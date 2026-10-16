// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ActivationBackwardParams, AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams,
    AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams, Conv1dParams,
    Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FRAMES_WORKGROUP, FmaParams, FramesParams,
    FusedResidualLnParams, FusedResidualLnTeeParams, FusedResidualRmsNormParams, FusedSwiGLUParams,
    GatedDeltaNetParams, GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams,
    GatherBwdParams, GatherParams, GroupedMatmulParams, GruParams, Kernel, LayerNorm2dParams,
    LayerNormBwdParams, LayerNormParams, Mamba2Params, MatmulParams, MatmulQkvParams,
    NarrowConcatParams, Pool1dParams, Pool2dParams, Pool3dParams, ReduceParams, ReverseParams,
    RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams, SampleParams, ScatterAddParams,
    SceBwdParams, SceParams, SelectiveScanParams, SoftmaxParams, TopKParams, TransposeParams,
    UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams, activation_backward_kernel,
    ada_layer_norm_backward_kernel, ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel,
    attention_decode_eligible, attention_decode_kernel, attention_kernel, attention_mask_kernel,
    axial_rope2d_kernel, batch_elementwise_region_kernel, batch_norm_inference_bwd_beta_kernel,
    batch_norm_inference_bwd_gamma_kernel, batch_norm_inference_bwd_input_kernel,
    batch_norm_inference_kernel, binary_c64_kernel, binary_kernel, cast_f32_to_f16_kernel,
    cast_kernel, compare_kernel, complex_cast_kernel, complex_norm_sq_backward_kernel,
    complex_norm_sq_kernel, concat_kernel, conjugate_c64_kernel, conv_transpose2d_kernel,
    conv_transpose3d_kernel, conv1d_kernel, conv1d_tiled_kernel, conv2d_kernel,
    conv3d_backward_input_kernel, conv3d_backward_weight_kernel, conv3d_ctile_kernel,
    conv3d_kernel, copy_kernel, copy_sanitize_kernel, cum_scan_kernel, cumsum_backward_kernel,
    cumsum_kernel, dequant_matmul_kernel, dequant_matmul_mlx_kernel, dequantize_i8_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel,
    fake_quantize_fixed_kernel, fake_quantize_perbatch_kernel, fft_butterfly_stage_kernel,
    fma_kernel, frames_kernel, fused_conv_bias_act_kernel, fused_residual_ln_kernel,
    fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel, fused_swiglu_kernel,
    gated_delta_net_kernel, gated_residual_backward_kernel, gated_residual_kernel,
    gather_axis_kernel, gather_backward_acc_kernel, gather_backward_zero_kernel,
    gather_elements_kernel, gather_kernel, gather_nd_kernel, gather_split_kernel,
    group_norm_backward_beta_kernel, group_norm_backward_gamma_kernel,
    group_norm_backward_input_kernel, group_norm_kernel, grouped_matmul_kernel, gru_kernel,
    im2col2d_kernel, layer_norm_backward_gamma_partial_kernel,
    layer_norm_backward_gamma_reduce_kernel, layer_norm_backward_input_kernel, layer_norm2d_kernel,
    layernorm_kernel, lstm_kernel, mamba2_kernel, matmul_bf16w_kernel,
    matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, maxpool2d_backward_kernel,
    maxpool3d_backward_kernel, narrow_kernel, note_attention_decomposed_route,
    note_attention_route, pool1d_kernel, pool2d_kernel, pool3d_kernel, q_conv2d_kernel,
    q_matmul_kernel, quantize_i8_kernel, reduce_kernel, reverse_kernel, rms_norm_backward_kernel,
    rms_norm_backward_param_kernel, rnn_kernel, rope_backward_kernel, rope_kernel, sample_kernel,
    scaled_dequantize_kernel, scaled_grouped_matmul_decode_kernel, scaled_matmul_decode_kernel,
    scaled_quant_scale_kernel, scaled_quantize_kernel, scatter_add_kernel, scatter_elements_kernel,
    scatter_nd_reduce_kernel, selective_scan_kernel, softmax_cross_entropy_backward_kernel,
    softmax_cross_entropy_kernel, softmax_cross_entropy_with_logits_kernel, softmax_kernel,
    topk_kernel, transpose_kernel, umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel,
    welch_peaks_gpu_kernel, where_kernel,
};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    /// The run path's join decision for `step`: `true` when this run encodes the
    /// call's dispatches into the enclosing batch, so its work rides the batch's
    /// submission instead of one of its own.
    ///
    /// That is the route ladder's own answer for the call — the two device-only
    /// [`crate::gguf_gpu::DequantMatmulPath`]s, whose entry point
    /// [`crate::gguf_gpu::encode_dequant_matmul_gguf_into`] can be handed a caller's
    /// encoder: the tiled one-row arm
    /// ([`crate::gguf_gpu::DequantMatmulPath::OneRowDevice`]) and the row-grouped
    /// GEMM above it
    /// ([`crate::gguf_gpu::DequantMatmulPath::GroupedDevice`]), which carries the
    /// tiled K-quants' multi-row calls and Q8_0's — the latter over that scheme's
    /// own 32-element blocks, the tile itself being only the K-quants' gate. Whether
    /// this *run* batches at all is the caller's business — the step-by-step and
    /// snapshot modes keep one operation per submission on purpose
    /// (`RLX_WGPU_ONE_OP_PER_PASS`, and the snapshot stops between steps), so both of
    /// them apply that mode to this answer rather than fold it in here.
    ///
    /// The answer is the ladder's, not the encode's: such a call can still be
    /// carried into an encode that refuses it — for Q8_0, whose per-row loop carries
    /// its one row alone, the refusal is recomputed by the entry point that owns it,
    /// [`crate::gguf_host`] included where a sharded arena or a parked weight lands
    /// it there — and the join branch owns that refusal (see its comment).
    ///
    /// `scale` is the step arm's own active-extent scaling, so the answer is for the
    /// row count in use, not the compiled one. A false answer is conservative: it
    /// covers every route that can read the host, and the caller is the join branch,
    /// which re-matches the step geometry it needs.
    fn joined_dequant_gguf(&self, step: &Step, scale: &impl Fn(u32) -> u32) -> bool {
        let Step::DequantMatmulGguf {
            m,
            k,
            scheme_id,
            x_byte_off,
            ..
        } = step
        else {
            return false;
        };
        crate::gguf_gpu::select_path(
            scale(*m) as usize,
            *k as usize,
            *scheme_id,
            *x_byte_off as usize,
            self.dequant_scratch_off,
        )
        .is_provably_device_only()
    }

    /// Whether this run treats `self.schedule[step_i]` as a host boundary: every
    /// host-classified step is one except a device-only call this run joins into
    /// the enclosing batch — a provably device-only packed multiply
    /// ([`Self::joined_dequant_gguf`]) or a buffer copy whose route answer is the
    /// device one ([`CopyRoute::joins_batch`]). The run path's scheduling decisions
    /// read this rather than the plan-time classification, which stays as it is and
    /// keeps its consumers (the uniform/bind-group lanes, the memory plan, the
    /// browser answer).
    ///
    /// A step this run *skips* never joins, whichever classification it carries: a
    /// baked static pack (`static_once_done`) is consumed by the pass loop or by the
    /// skip loop that follows the batch, so it is not work this boundary can hand to
    /// the enclosing run. That leaves each skip where it was before the copy join —
    /// a host-classified staging copy keeps this boundary's `true` (the skip loop
    /// consumes it), a device one keeps `false` (the pass loop consumes it together
    /// with its bind-group cursor).
    ///
    /// `step_i` must be within `self.schedule`; every caller checks that first.
    ///
    /// The sites that read the answer, so "the boundary stays correct everywhere it
    /// is consulted" has one list:
    /// * **a joined call is device work** at `starting_on_host`, at the post-batch
    ///   `need_host_sync` and its mirror clear, and at the door in front of device
    ///   work after a host step (which now fires for it too). These are the sites
    ///   that act on this step only because the plan classifies it as a host step.
    /// * **a joined call is device work *inside the batch*** for
    ///   `step_needs_pass_flush`, where it can now be the `prev` step: on
    ///   Vulkan/DX12-class adapters that arm asks for the pass split and the drain
    ///   the copy's own submission used to provide (commented at that arm).
    /// * **the run's exit path**: a joined call keeps the batch carried, so the loop
    ///   runs on to the tail that reads the outputs back instead of ending the run
    ///   after the arm — which tail a copy-ending graph leaves through is decided
    ///   here, and that is why every such tail runs the debug scans
    ///   ([`Self::debug_scans_after_run`]).
    /// * **untouched**: the lane cursors (a joined call consumes no uniform,
    ///   bind-group or meta-buffer slot, so no later step's index moves), the pass
    ///   loop's `break` on a host step and its two `static_once` skips, the final
    ///   mirror flush, the host route's own readback and drain, `bypasses_host_cache`
    ///   (a call the join consumed never reaches it, but one whose encode refused it
    ///   does — `DequantMatmulGguf` is not a cache-aware arm, so that tail drops the
    ///   mirror after the refusal's recompute), and the `RLX_WGPU_DBG_COUNTERS` census.
    fn step_breaks_batch(
        &self,
        step_i: usize,
        scale: &impl Fn(u32) -> u32,
        one_op_per_pass: bool,
        host_cache: &rlx_gpu_host::HostTensorCache,
        dev: &crate::device::WgpuDevice,
    ) -> bool {
        let step = &self.schedule[step_i];
        // A step this run skips is not work this boundary can hand to the
        // enclosing run — see the doc above.
        let skipped = self.static_once_done && self.static_once_steps.contains(&step_i);
        let joins = !one_op_per_pass
            && !skipped
            && (self.joined_dequant_gguf(step, scale)
                || self
                    .buffer_copy_route_now(step, host_cache, dev)
                    .is_some_and(CopyRoute::joins_batch));
        step_is_tail_host(step) || (step_runs_on_host(step) && !joins)
    }

    /// The run path's route answer for a copy `step`, or `None` for every other
    /// step kind. **This is the one [`buffer_copy_route`] call the run path makes for
    /// a copy**: the batch boundary asks it (through [`CopyRoute::joins_batch`]) and
    /// the host arm routes by it, both from the same run state — so the boundary and
    /// the arm cannot disagree, which is what makes a join safe to take.
    ///
    /// `deferred_writes` is the mirror's own answer, so a copy with a pending host
    /// write can never answer `Device`: the host route is the one that reads the
    /// mirror, and joining a call that could reach it would run host work with an
    /// encoder carried.
    fn buffer_copy_route_now(
        &self,
        step: &Step,
        host_cache: &rlx_gpu_host::HostTensorCache,
        dev: &crate::device::WgpuDevice,
    ) -> Option<CopyRoute> {
        let Step::BufferCopy {
            src_byte_off,
            dst_byte_off,
            bytes,
        } = step
        else {
            return None;
        };
        Some(buffer_copy_route(&CopyRouteInputs {
            src: *src_byte_off,
            dst: *dst_byte_off,
            nbytes: *bytes as u64,
            arena_size: self.arena.size as u64,
            max_binding: dev.device.limits().max_storage_buffer_binding_size,
            // `shard_size` is 0 exactly when the arena is unsharded.
            shard_size: self.arena.shard_size as u64,
            weight_operand: crate::buffer::is_weight_off(*src_byte_off as usize)
                || crate::buffer::is_weight_off(*dst_byte_off as usize),
            deferred_writes: host_cache.has_deferred_writes(),
            force_host: rlx_ir::env::flag("RLX_WGPU_HOST_BUFFER_COPY"),
        }))
    }

    /// Encode the whole device-side body of a `Step::BufferCopy` into `enc`: one
    /// `copy_kernel` dispatch over the storage window `[base, base + size)` that
    /// [`buffer_copy_route`] picked, in a compute pass of its own.
    ///
    /// Shared by the two places the copy's work is encoded — the host arm, which
    /// submits the encoder it creates, and the join branch, which hands the
    /// enclosing batch's encoder — so both encode the identical dispatch. Its own
    /// pass is what keeps the barrier the arena's aliased slots need between this
    /// write and every later read of it: a pass of its own is exactly what the
    /// unjoined arm's submission of its own used to provide. Nothing is staged for
    /// it: the window is the arena itself, and the per-call uniform and bind group
    /// live as long as the encoder that references them.
    fn encode_arena_copy_dispatch(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src: u64,
        dst: u64,
        nbytes: u64,
        base: u64,
        size: u64,
    ) {
        // `copy_route` refused a zero-byte copy above, and every producer sizes a
        // slot at `elems * 4`, so the element count below is exact.
        let elems = (nbytes / 4) as u32;
        let p = CopyParams {
            n: elems,
            in_off: (src.saturating_sub(base) / 4) as u32,
            out_off: (dst.saturating_sub(base) / 4) as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            _p4: 0,
        };
        let ck = copy_kernel(&dev.device);
        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu arena_copy uniform"),
            size: std::mem::size_of::<CopyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = bind_arena_window(&dev.device, ck, &self.arena, base, size, &u);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu arena_copy pass"),
            ..Default::default()
        });
        pass.set_pipeline(&ck.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(elems, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }

    /// Register a resident-KV row feed: `handle_name` (a device-resident input,
    /// see `bind_gpu_handle`) receives row data from graph output `output_index`.
    /// Answers whether this executable can bind that name — the same question
    /// [`Self::bind_gpu_handle`] asks, so a name that can be bound can be
    /// registered; `false` means the registration is not there to be fed.
    ///
    /// Pairs with `Op::KvAppend`. Without residency the op does not actually
    /// remove the O(context) cost — the concat it replaces copies the cache
    /// on-GPU, but the runtime still re-uploads the past KV as a graph input
    /// every step, so the cost merely moves. This is the half that keeps the
    /// cache on the device between steps.
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        if !self.can_bind_gpu_handle(handle_name) {
            return false;
        }
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
        true
    }

    /// Fold each registered feed's new-token row (`src_row` of its output) into
    /// the resident handle at `dst_row`, device-side.
    ///
    /// Mirrors `rlx-metal`'s semantics, including the aliased case: when the
    /// output IS the input (in-place `KvAppend`, `out_id == in_id`), a copy with
    /// `src_row != dst_row` is a legitimate intra-buffer relocate — the graph
    /// writes the new token at a fixed row and this moves it to the growing
    /// `past_seq` row. Only a genuine no-op (same node AND same row) is skipped,
    /// and it stays a success: a consumer that folds a row where it already sits
    /// must not be pushed onto its host fallback.
    ///
    /// Every registered name is resolved and vetted BEFORE any byte moves, so one
    /// name that cannot take its row fails the whole call and leaves every cache
    /// reading back as it did before the call — `ExecutableGraph::feed_kv_row`
    /// holds the list of what makes a name unservable.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) -> bool {
        let Some(dev) = crate::device::wgpu_device() else {
            return false;
        };
        if self.kv_row_feeds.is_empty() || row_elems == 0 {
            return false;
        }
        // Vet every registered feed first: reporting failure after some row had
        // already moved would leave some caches current and others not.
        let mut moves: Vec<(String, NodeId, NodeId)> = Vec::with_capacity(self.kv_row_feeds.len());
        for (name, &out_idx) in &self.kv_row_feeds {
            let Some(&out_id) = self.graph.outputs.get(out_idx) else {
                return false;
            };
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                return false;
            };
            // "Held here": this name's rows are in the arena slot (authoritative
            // once resident), or in the host mirror this executable still holds.
            // A clone emptied the mirror when the rows became resident, so it
            // holds neither and reports failure rather than feeding a zeroed
            // cache.
            let held = self.gpu_handle_resident.contains(name)
                || self.gpu_handles.get(name).is_some_and(|m| !m.is_empty());
            if !held || !self.arena.has(in_id) || !self.arena.has(out_id) {
                return false;
            }
            // A row past what its node declares is refused rather than copied from
            // or into whatever the arena holds beside the slots. A divide, not a
            // product: release has no overflow checks (`apple` is release with
            // `panic = "abort"`), so a huge row index would wrap the product.
            if dst_row >= self.extent_elems(in_id) / row_elems
                || src_row >= self.extent_elems(out_id) / row_elems
            {
                return false;
            }
            moves.push((name.clone(), in_id, out_id));
        }
        let bytes = row_elems * std::mem::size_of::<f32>();
        for (name, in_id, out_id) in moves {
            if in_id != out_id || src_row != dst_row {
                let staging = self.kv_feed_staging.get_or_insert_with(|| {
                    dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu kv feed staging"),
                        size: bytes as u64,
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });
                // Grow if a later feed carries a wider row.
                if staging.size() < bytes as u64 {
                    *staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu kv feed staging"),
                        size: bytes as u64,
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                self.arena.copy_within_device(
                    &dev.device,
                    &dev.queue,
                    staging,
                    self.arena.offset(out_id) + src_row * bytes,
                    self.arena.offset(in_id) + dst_row * bytes,
                    bytes,
                );
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name, Vec::new());
        }
        true
    }

    /// How many schedule steps of each kind this executable will dispatch.
    ///
    /// wgpu has no equivalent of `rlx-metal`'s per-thunk profile, so there was
    /// no way to see the shape of a lowered graph short of a debugger. Counts
    /// are exact and machine-independent, which makes them usable on a
    /// contended box where timings are not: "111 dispatches became 56" is a
    /// fact about the schedule, not about the machine that ran it.
    pub fn step_kind_counts(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut m = std::collections::BTreeMap::new();
        for s in &self.schedule {
            *m.entry(crate::backend::step::step_name(s)).or_insert(0) += 1;
        }
        m
    }

    /// `(steps marked static-once, skip armed)` — diagnostics for the
    /// weight-pack skip.
    ///
    /// A `Concat`/`Expand`/`Cast` over `Param`s (the fused QKV and gate+up packs
    /// the matmul-fusion passes build) is invariant across `run()`s, so lowering
    /// marks its steps and later runs skip them. On a Llama decode those packs
    /// are close to half of all DRAM traffic, so whether the skip actually arms
    /// is worth being able to assert rather than infer.
    ///
    /// The two halves are independent: lowering can mark steps (`.0 > 0`) while
    /// no executed path ever sets the flag (`.1 == false`), in which case the
    /// optimisation is dead and the packs are rebuilt every run.
    pub fn static_once_report(&self) -> (usize, bool) {
        (self.static_once_steps.len(), self.static_once_done)
    }

    /// Arm the static-weight-pack skip: the schedule has run, so every pack has
    /// been materialised and later runs may skip re-computing it.
    ///
    /// Called from EVERY readback return in `run_inner`, not just some. Two of
    /// the five did this inline and three did not, so whether the optimisation
    /// engaged depended on which readback path the graph happened to take —
    /// a graph with small outputs returns via the "tiny" path and never armed,
    /// silently rebuilding its constant weight packs on every run. Routing all
    /// five through one helper is what keeps that from drifting apart again.
    fn arm_static_once(&mut self) {
        if !rlx_compile::memory::static_weight_pack_skip_enabled() {
            return;
        }
        if !self.static_once_done && !self.static_once_steps.is_empty() {
            self.static_once_done = true;
        }
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        if rlx_ir::env::flag("RLX_WGPU_SNAPSHOT") && !self.in_snapshot {
            self.snapshot_nodes(inputs);
        }
        self.pending_read_indices = read_indices.map(|s| s.to_vec());
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        outs
    }

    /// Print each node's output **at the moment it is produced**.
    ///
    /// Every other way of inspecting a node is unsound. Adding it as a graph
    /// output changes the compilation. Reading the arena after the run reports a
    /// reused slot's final occupant, and cannot tell an elided node from one a
    /// kernel wrongly zeroed. Both produced confident wrong answers on the
    /// `rlx-vieeg` wgpu divergence.
    ///
    /// This replays the *same* compiled schedule truncated to its first `k`
    /// steps, for every `k`, and reads the node that step `k-1` produced. Bind
    /// groups, uniforms and the arena are untouched — the prefix indices line up
    /// with the full run — so the value read is the one the graph really
    /// computed at that point. Cost is O(n²) dispatches, which is fine for a
    /// debug path on a few hundred nodes.
    ///
    /// `RLX_WGPU_SNAPSHOT=1`.
    fn snapshot_nodes(&mut self, inputs: &[(&str, &[f32])]) {
        let Some(dev) = crate::device::wgpu_device() else {
            return;
        };
        let total = self.schedule.len();
        eprintln!("[wgpu-snapshot] replaying {total} steps, reading each node as it is produced");
        self.in_snapshot = true;
        for k in 1..=total {
            // `Step` is not `Clone`, so stop the loop by index rather than by
            // rebuilding a truncated schedule — which also keeps the bind-group
            // and uniform indices identical to the full run.
            self.snapshot_stop = Some(k);
            self.dispatch_only = true;
            let _ = self.run_inner(inputs);
            self.dispatch_only = false;
            let Some(&node) = self.step_nodes.get(k - 1) else {
                continue;
            };
            // A step whose node is also produced by a later step (fused tees
            // write more than one slot) is reported at each point it is written.
            if !self.arena.has(node) {
                continue;
            }
            let data = self.arena.read_f32(&dev.device, &dev.queue, node);
            let max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let nz = data.iter().filter(|&&v| v != 0.0).count();
            // For a `[1, N, D]` activation, report the final row on its own.
            // A fault confined to the last token — TinyMyo's shape — shows up as
            // this ratio departing from the rest of the tensor, which needs no
            // reference run and no cross-backend node-id alignment (node ids do
            // not correspond across backends, since each fuses differently).
            let dims: Vec<usize> = self
                .graph
                .node(node)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            // Max over the LAST row, by the same rule the CPU dump uses
            // (`RLX_CPU_DUMP_NODES`), so the two are directly comparable. The
            // rank-3-only block below stays for the tail/rest ratio.
            let row = dims.last().copied().unwrap_or(data.len()).max(1);
            let sum: f64 = data.iter().map(|&v| v as f64).sum();
            let tail_row = data
                .len()
                .checked_sub(row)
                .map(|o| data[o..].iter().fold(0.0f32, |m, &v| m.max(v.abs())))
                .unwrap_or(max);
            let tabs: f64 = data
                .len()
                .checked_sub(row)
                .map(|o| data[o..].iter().map(|&v| v.abs() as f64).sum())
                .unwrap_or(0.0);
            // `RLX_WGPU_ROW_TABS=<step>`: per-row sum of |v| for one step's node,
            // to diff against the CPU dump's `RLX_CPU_ROW_TABS` and see exactly
            // which rows a kernel gets wrong.
            if rlx_ir::env::parse_or::<usize>("RLX_WGPU_ROW_TABS", usize::MAX) == k - 1 {
                for (r, chunk) in data.chunks(row).enumerate() {
                    let t: f64 = chunk.iter().map(|&v| v.abs() as f64).sum();
                    eprintln!("  rowtabs step={} row={r} tabs={t:.6}", k - 1);
                }
            }
            let tail = if dims.len() == 3
                && dims[0] == 1
                && dims[1] > 1
                && data.len() == dims[1] * dims[2]
            {
                let d = dims[2];
                let last = &data[data.len() - d..];
                let rest = &data[..data.len() - d];
                let lm = last.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let rm = rest.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-12);
                Some(format!("  tail_max={lm:.6} tail/rest={:.4}", lm / rm))
            } else {
                None
            };
            // `RLX_WGPU_SNAPSHOT_WATCH=<id>` also reports one chosen node after
            // every step. A fused op can write a slot it does not own — the tee
            // form of residual+LayerNorm does — so a node with no step of its own
            // may still be filled, and the only way to see *when* is to watch it.
            let watch = rlx_ir::env::var("RLX_WGPU_SNAPSHOT_WATCH")
                .and_then(|v| v.parse::<u32>().ok())
                .map(rlx_ir::NodeId)
                .filter(|w| self.arena.has(*w));
            let watched = watch.map(|w| {
                let d = self.arena.read_f32(&dev.device, &dev.queue, w);
                let mx = d.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let n = d.iter().filter(|&&v| v != 0.0).count();
                format!("   watch[{}] max={mx:.6} nz={n}/{}", w.0, d.len())
            });
            eprintln!(
                "  step {:>3} node {:>4} {:<34} max={max:.6} tail={tail_row:.6} sum={sum:.6} tabs={tabs:.6} nonzero={nz}/{}{}",
                k - 1,
                node.0,
                format!("{:?}", self.graph.node(node).op.kind()),
                data.len(),
                watched.unwrap_or_default()
            );
            if let Some(t) = tail {
                eprintln!("{t}");
            }
        }
        self.snapshot_stop = None;
        self.in_snapshot = false;
    }

    /// Async sibling of [`Self::run`] for the browser, where GPU→CPU readback
    /// cannot block the event loop. All compute is dispatched + submitted
    /// synchronously (via the normal `run_inner` in dispatch-only mode); the
    /// outputs are then read back from the arena asynchronously.
    ///
    /// Supports pure feed-forward graphs only — graphs with host-executed ops
    /// (GGUF dequant, LSTM/GRU, GatedDeltaNet, FFT-host, …) map intermediate
    /// results back mid-graph, which would block the browser. Such graphs
    /// panic with a clear message rather than hang.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        assert!(
            !self
                .schedule
                .iter()
                .any(|s| step_runs_on_host(s) || step_is_tail_host(s)),
            "rlx-wgpu run_async: graph contains host-executed ops unsupported on wasm"
        );

        // 1. Dispatch + submit all compute, skipping the blocking readback.
        self.dispatch_only = true;
        let _ = self.run_inner(inputs);
        self.dispatch_only = false;

        let dev =
            wgpu_device().expect("rlx-wgpu: device not initialized (call init_wgpu_device first)");

        // 2. Read the selected outputs back from the arena asynchronously.
        let plan = self.readback_plan();
        let out_ids_all: Vec<_> = self.graph.outputs.clone();
        let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
        let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
        if layout.regions.is_empty() {
            return Vec::new();
        }
        ReadbackStaging::prepare(&dev.device, &mut self.readback_staging, layout.total_bytes);
        let staging_buf = self
            .readback_staging
            .as_ref()
            .expect("readback staging")
            .buffer()
            .clone();

        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu async readback"),
            });
        encode_readback_copies(&mut enc, &self.arena, &staging_buf, &out_ids, &layout);
        crate::backend::note_device_submit();
        dev.queue.submit(std::iter::once(enc.finish()));

        crate::buffer::wasm_async::map_read_async(&staging_buf, layout.total_bytes)
            .await
            .expect("rlx-wgpu async buffer map failed");

        let partial = decode_mapped_readback_f32(&staging_buf, &layout);
        self.pack_readback_outputs(&plan, partial)
    }

    /// True when the compiled schedule contains host-executed ops (GGUF
    /// dequant, LSTM/GRU, collectives, FFT-host, …). Such graphs cannot
    /// run on browser WebGPU (`run_async` rejects them).
    ///
    /// Read off the plan-time classification, deliberately: it stays pessimistic,
    /// and stays correct, because the host arms remain reachable in the run-time
    /// route ladder — a packed multiply and a buffer copy whose route answers
    /// `Device` are joined by a run-time decision, and either runs on the host
    /// whenever its own ladder answer is not device-only.
    pub fn requires_host_execution(&self) -> bool {
        self.schedule
            .iter()
            .any(|s| super::step_runs_on_host(s) || super::step_is_tail_host(s))
    }

    pub(crate) fn run_tail_host_audio_ops(&self, dev: &crate::device::WgpuDevice) {
        if !self.schedule.iter().any(step_is_tail_host) {
            return;
        }
        for step in &self.schedule {
            if !step_is_tail_host(step) {
                continue;
            }
            match step {
                Step::WelchPeaksHost {
                    spec_byte_off,
                    dst_byte_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => {
                    crate::welch_peaks_host::run_welch_peaks(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *dst_byte_off as usize,
                        *welch_batch as usize,
                        *n_fft as usize,
                        *n_segments as usize,
                        *k as usize,
                    );
                }
                Step::LogMelHost {
                    spec_byte_off,
                    filt_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                Step::LogMelBackwardHost {
                    spec_byte_off,
                    filt_byte_off,
                    dy_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dy_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                _ => {}
            }
        }
    }

    pub(crate) fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut report = RunReport::start(self.in_snapshot);
        // Lazy compile path: if we deferred compile waiting for shapes,
        // infer the binding from input data lengths now and compile.
        if self.unresolved.is_some() {
            report.open(Phase::Compile, 0, &self.schedule);
            self.lazy_compile_for_inputs(inputs);
        }
        report.open(Phase::Stage, 0, &self.schedule);
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        self.stage_gpu_handle_inputs(dev, inputs);
        // Always re-stage graph inputs. Arena liveness reuse may overwrite
        // input slots once the input dies mid-graph; a hash-based skip across
        // runs would leave those slots holding stale activations (empty/garbage
        // outputs on Conformer-CTC and similar). `RLX_WGPU_FORCE_INPUT_UPLOAD`
        // is retained as a documented no-op for callers that set it.
        let _ = rlx_ir::env::flag("RLX_WGPU_FORCE_INPUT_UPLOAD");
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                self.arena.write_f32(&dev.queue, id, data);
            }
        }
        for &(act_id, act, ref src_name) in &self.coop_f16_host_activations {
            let src =
                host_tensor_f32(src_name, inputs, &self.stashed_params).unwrap_or_else(|| {
                    panic!("rlx-wgpu CoopF16Vk host activation: missing tensor {src_name:?}")
                });
            let mirrored = apply_activation_host(act, src);
            self.arena.write_f32(&dev.queue, act_id, &mirrored);
        }
        if !self.coop_f16_host_activations.is_empty() {
            // Ensure host staging writes are visible before CoopF16Vk reads f16.
            let flush = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rlx-wgpu host mirror flush"),
                });
            crate::backend::note_device_submit();
            dev.queue.submit(std::iter::once(flush.finish()));
            crate::backend::note_device_wait();
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Active-extent (PLAN L1): scale safe Steps' primary dim by
        // actual/upper. Used in BOTH the uniform-write loop (so the
        // kernel sees the scaled count) AND the dispatch loop (so the
        // workgroup grid is shrunk). Off entirely when any step vetoes it.
        let active = self.active_extent_in_effect();
        if rlx_ir::env::flag("RLX_WGPU_ACTIVE_TRACE")
            && self.active_extent.is_some()
            && active.is_none()
        {
            let bad: Vec<_> = self
                .schedule
                .iter()
                .filter(|s| !s.safe_for_active_extent())
                .map(step_name)
                .collect();
            eprintln!(
                "[wgpu-active] extent={:?} DISABLED by unsafe steps: {bad:?}",
                self.active_extent
            );
        }
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // Stage uniform writes — but skip the loop entirely when the
        // bytes already in the uniforms match this run's active extent.
        // BERT inference at fixed batch hits this path: 100+ tiny
        // queue.write_buffer calls (one per Step) collapse to zero,
        // saving milliseconds of staging-copy overhead.
        let need_uniform_writes = self.uniforms_active_extent != Some(active);
        if need_uniform_writes {
            let mut gpu_ui = 0usize;
            for (step_i, step) in self.schedule.iter().enumerate() {
                if step_runs_on_host(step) {
                    continue;
                }
                if self.static_once_done && self.static_once_steps.contains(&step_i) {
                    if !matches!(step, Step::FftGpu { .. }) {
                        gpu_ui += 1;
                    }
                    continue;
                }
                if rlx_ir::env::var("RLX_DBG_STEP").is_some() {
                    let usz = self.uniforms.get(gpu_ui).map(|u| u.size()).unwrap_or(0);
                    eprintln!("[wgpu-uni gpu_ui={gpu_ui} ubuf={usz}B] {}", step_name(step));
                }
                report.note_step(step);
                match step {
                    Step::CastF32ToF16 { .. } => {
                        // Params are static for this step (offset+len), so the
                        // pre-pass write at compile time is sufficient. No
                        // active-extent scaling — len is the full element count.
                    }
                    Step::IndexingNd { params, .. } => {
                        // No active-extent scaling (see
                        // `safe_for_active_extent`); write the params as planned.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::ScaledLowp { params, .. } => {
                        // Same: shapes are baked, nothing to rescale.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::QuantI8 { params, .. } => {
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::QMatMul { params, .. } => {
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::QConv2d { params, .. } => {
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::BatchNormInference { params, .. } => {
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::Matmul {
                        m,
                        k,
                        n,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32,
                        batch,
                        a_batch_stride,
                        b_batch_stride,
                        c_batch_stride,
                        has_bias,
                        bias_off_f32,
                        act_id,
                        ..
                    } => {
                        // PLAN L1 (safe at any batch — c_batch_stride is
                        // pre-baked at compile time at FULL m, so scaling
                        // params.m only changes per-thread bound checks).
                        // `scale` is the identity on a schedule carrying the 1×1
                        // pointwise-conv GEMM (`from_conv1x1`): that step makes
                        // `all_safe_for_active` false, so `active` is None throughout.
                        let m_scaled = scale(*m);
                        let p = MatmulParams {
                            m: m_scaled,
                            k: *k,
                            n: *n,
                            a_off: *a_off_f32,
                            b_off: *b_off_f32,
                            c_off: *c_off_f32,
                            batch: *batch,
                            a_batch_stride: *a_batch_stride,
                            b_batch_stride: *b_batch_stride,
                            c_batch_stride: *c_batch_stride,
                            has_bias: *has_bias,
                            bias_off: *bias_off_f32,
                            act_id: *act_id,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Binary { params } | Step::Compare { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Unary { params, .. } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Where { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Fma { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ReluBackward { params } | Step::ActivationBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Reduce { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Softmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SoftmaxCrossEntropy { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AttentionMask { params } => {
                        // Not active-extent scaled — see
                        // `Step::safe_for_active_extent`, which vetoes that mode
                        // for the whole graph — so the params go out verbatim and
                        // the rows are the planned ones.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::SoftmaxCrossEntropyWithLogits { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SoftmaxCrossEntropyBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupNorm { params } => {
                        // NCHW geometry is fixed; there is no sequence extent to
                        // scale, so the params go out verbatim.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::LayerNorm2d { params } => {
                        // Same as GroupNorm: NCHW geometry is fixed and the
                        // step is never scaled, so the params go out verbatim.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::FusedSwiGLU { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RmsNormBackwardInput { params }
                    | Step::RmsNormBackwardGamma { params }
                    | Step::RmsNormBackwardBeta { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardInput { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaPartial { params, .. } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaReduce { params } => {
                        // `outer` here is the partial chunk count (not
                        // a batch dim) — do NOT apply active-extent
                        // scaling.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::CumsumBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RopeBackward { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Cumsum { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::CumScan { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FftGpu { .. } => {}
                    Step::Copy { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Cast { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ComplexCast { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BinaryC64 { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ComplexNormSq { params }
                    | Step::ComplexNormSqBackward { params }
                    | Step::ConjugateC64 { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FftButterflyStage { params } => {
                        let mut p = *params;
                        p.batch = scale(p.batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BufferCopy { .. } => {}
                    Step::ElementwiseRegion { params } => {
                        // Active-extent: scale element count.
                        let mut p = *params;
                        p.len = scale(p.len);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BatchElementwiseRegion { params } => {
                        let mut p = *params;
                        p.slice_len = scale(p.slice_len);
                        p.num_batch = scale(p.num_batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Transpose { params, .. } => {
                        // PLAN L1: when bucket_outermost == 1, scale
                        // `out_total` proportional to scaling `out_dim_0`.
                        // Other transposes leave out_total at full extent
                        // (predicate prevents the active-extent path).
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Narrow { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Concat { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Gather { params } => {
                        let mut p = *params;
                        p.n_out = scale(p.n_out);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherAxis { params } => {
                        let mut p = *params;
                        // Active-extent trims the LEADING (outer) dim only; the
                        // gathered axis + trailing are not seq-proportional.
                        // Scaling `total` directly would drop trailing lanes for
                        // a seq-collapsing gather (e.g. last-token select, outer=1).
                        p.total = scale(p.outer) * p.num_idx * p.trailing;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Attention { params, .. } => {
                        // PLAN L1: scale seq_q + seq_k. Stride fields
                        // (seq_q_stride / seq_k_stride) stay at the
                        // compile-time full extent, so per-(batch, head)
                        // offset math in the WGSL stays correct.
                        let mut p = *params;
                        p.seq_q = scale(p.seq_q);
                        p.seq_k = scale(p.seq_k);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AttentionBackward { params, .. } => {
                        let mut p = *params;
                        if p.wrt == 0 {
                            p.seq_q = scale(p.seq_q);
                        } else {
                            p.seq_k = scale(p.seq_k);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Rope { params } => {
                        // PLAN L1: scale `seq` and `n_total` proportionally.
                        // `seq_stride` and `batch` stay at compile-time
                        // values; the WGSL kernel uses them for buffer
                        // offsets while `seq` / `n_total` are loop bounds.
                        let mut p = *params;
                        let s_active = scale(p.seq);
                        p.seq = s_active;
                        p.n_total = p.batch * s_active * p.last_dim;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Expand { params, .. } => {
                        // PLAN L1: same pattern as Transpose.
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Reverse { params, .. } => {
                        // PLAN L1: same per-instance decision as Expand — scale
                        // the leading dim only when this flip does not touch it.
                        let mut p = *params;
                        if p.scale_leading == 1 && p.dim_0 > 0 {
                            let scaled_d0 = scale(p.dim_0);
                            let inner = p.total / p.dim_0;
                            p.total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Argmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool2d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MaxPool2dBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MaxPool3dBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3dBackwardInput { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3dBackwardWeight { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupNormBackwardInput { params }
                    | Step::GroupNormBackwardGamma { params }
                    | Step::GroupNormBackwardBeta { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AxialRope2d { params } => {
                        let mut p = *params;
                        p.batch = scale(p.batch);
                        p.n_total = p.batch * p.seq * p.hidden;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FakeQuantizeFixed { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        // axis=None: inner tracks full extent with n.
                        if p.chan_dim <= 1 {
                            p.inner = p.n.max(1);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FakeQuantizePerBatch { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        if p.chan_dim <= 1 {
                            p.inner = p.n.max(1);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv2d { params } | Step::Conv2dTiled { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedConvBiasAct { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ConvTranspose2d { params } | Step::ConvTranspose2dTiled { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ConvTranspose3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ScatterAdd { params } => {
                        // Two-phase: phase 0 zeros the FULL output (preserves
                        // accumulator semantics); phase 1 scatters first
                        // num_updates_active updates only.
                        let mut p = *params;
                        if p.op == 1 {
                            p.num_updates = scale(p.num_updates);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::TopK { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::WelchPeaksGpu { params } => {
                        let mut p = *params;
                        p.welch_batch = scale(p.welch_batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    // The one word the mode scales (see `safe_for_active_extent`).
                    Step::Frames { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::UmapKnn { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupedMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ScaledGroupedMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Sample { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SelectiveScan { params } => {
                        // Predicate-gated to batch=1: scale seq.
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Mamba2 { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Gru { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Rnn { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LstmGpu { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedDeltaNet {
                        params,
                        use_gpu: true,
                        ..
                    } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmulMlx { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    // Im2ColGpu params are static (written once at compile);
                    // no active-extent scaling.
                    Step::Im2ColGpu { .. }
                    | Step::GatherSplit { .. }
                    | Step::DequantMatmulGguf { .. }
                    | Step::DequantMatmulInt8Host { .. }
                    | Step::DequantMatmulMlxHost { .. }
                    | Step::Conv2dHost { .. }
                    | Step::DequantGroupedMatmulGguf { .. }
                    | Step::DequantGroupedMatmulMlxHost { .. }
                    | Step::GatedDeltaNet { use_gpu: false, .. }
                    | Step::Lstm { .. }
                    | Step::ConvTranspose2dHost { .. }
                    | Step::ConvTranspose3dHost { .. }
                    | Step::GroupNormHost { .. }
                    | Step::LayerNorm2dHost { .. }
                    | Step::ResizeNearest2xHost { .. }
                    | Step::ReverseHost { .. }
                    | Step::ArgReduceHost { .. }
                    | Step::AxialRope2dHost { .. }
                    | Step::GruHost { .. }
                    | Step::RnnHost { .. }
                    | Step::Llada2GroupLimitedGate { .. }
                    | Step::UmapKnnHost { .. }
                    | Step::MsDeformAttnHost { .. }
                    | Step::CollectiveHost { .. }
                    | Step::CustomHost { .. }
                    | Step::FftHost { .. }
                    | Step::FftQHost { .. }
                    | Step::ScanHost { .. }
                    | Step::HostOp { .. }
                    | Step::CpuIndexing { .. }
                    | Step::ConcatHost { .. }
                    | Step::ConcatHostPieces { .. }
                    | Step::TransposeHost { .. }
                    | Step::NarrowHost { .. }
                    | Step::ExpandHost { .. }
                    | Step::SpdHost { .. }
                    | Step::Im2ColHost { .. }
                    | Step::Conv2dBackwardWeightHost { .. }
                    | Step::Conv2dBackwardInputHost { .. }
                    | Step::RngNormalHost { .. }
                    | Step::RngUniformHost { .. }
                    | Step::WelchPeaksHost { .. }
                    | Step::LogMelHost { .. }
                    | Step::LogMelBackwardHost { .. } => {}
                    Step::FusedResidualLn { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualLnTee { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualRmsNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AdaLayerNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedResidual { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AdaLayerNormBackward { params } => {
                        let mut p = *params;
                        p.mod_rows = scale(p.mod_rows);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedResidualBackward { params } => {
                        let mut p = *params;
                        p.mod_rows = scale(p.mod_rows);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MatmulQkv { params, kind: _ } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    // Raw-GPU custom op: its storage params buffer is written at
                    // compile time (mapped_at_creation) with static, window-
                    // relative offsets — no active-extent scaling. No-op here so
                    // `gpu_ui` still advances for its uniform slot.
                    Step::WgpuGpuKernel { .. } => {}
                    #[cfg(feature = "splat")]
                    Step::GaussianSplatRender { .. }
                    | Step::GaussianSplatRenderBackward { .. }
                    | Step::GaussianSplatPrepare { .. }
                    | Step::GaussianSplatRasterize { .. } => {}
                }
                if !matches!(step, Step::FftGpu { .. }) {
                    gpu_ui += 1;
                }
            }
            if rlx_ir::env::flag("RLX_WGPU_DBG_COUNTERS") {
                let host = self
                    .schedule
                    .iter()
                    .filter(|s| step_runs_on_host(s))
                    .count();
                let tail = self
                    .schedule
                    .iter()
                    .filter(|s| step_is_tail_host(s))
                    .count();
                eprintln!(
                    "[wgpu-counters] steps={} host={host} tail_host={tail} uniforms={} \
                     bind_groups={} final_gpu_ui={gpu_ui}",
                    self.schedule.len(),
                    self.uniforms.len(),
                    self.bind_groups.len(),
                );
            }
            self.uniforms_active_extent = Some(active);
        }

        // Encode + submit.
        let mm_k = matmul_kernel(&dev.device);
        let mm_w_active = matmul_wide_active_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_bf16w = matmul_bf16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f16_vk = matmul_coop_f16_vulkan_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
        let mm_cast = cast_f32_to_f16_kernel(&dev.device);
        let bk = binary_kernel(&dev.device);
        let uk = unary_kernel(&dev.device);
        let ck = compare_kernel(&dev.device);
        let wk = where_kernel(&dev.device);
        let fk = fma_kernel(&dev.device);
        let abk = activation_backward_kernel(&dev.device);
        // One dispatch per compute pass ⇒ WebGPU inserts a full memory barrier
        // between every op, eliminating intra-pass hazard races on aliased arena
        // slots (the source of nondeterministic deep-graph results on wgpu).
        // The snapshot needs exactly one step per submitted pass so it can stop
        // after a chosen step and read what that step produced.
        let one_op_per_pass =
            rlx_ir::env::flag("RLX_WGPU_ONE_OP_PER_PASS") || self.snapshot_stop.is_some();
        let mut step_i = 0;
        let mut gpu_bi = 0usize;
        let mut fft_i = 0usize;
        let mut host_cache = rlx_gpu_host::HostTensorCache::new();
        // The batch's encoder. A joined call — a packed multiply or a
        // device-route copy — hands it to the next outer iteration instead of
        // submitting, so one contiguous run of device work is one command buffer
        // and one submission — the route's own `enc.finish()` would be exactly
        // the per-call submission this join removes. It also removes command
        // buffers rather than adding any: a batch creates its encoder once (the
        // loop below reuses it across joined calls), so a forward pass holds at
        // most one encoder that has been created but not yet submitted, against
        // the graphics layer's own budget of those (4096 on Metal). Never `Some`
        // when the loop exits: the loop condition keeps iterating until the
        // carried encoder has been submitted, and every other `break` out of the
        // loop happens after a submission — the snapshot break below is the one
        // that does not, and it cannot see a carried encoder because the snapshot
        // forces `one_op_per_pass`, which makes the join inert.
        let mut pending_enc: Option<wgpu::CommandEncoder> = None;
        // "This batch dispatched something" — set by any dispatched or joined
        // step, reset when the batch's encoder is created, and read by the
        // f16-mirror pass split and by the host-mirror invalidation.
        let mut pass_dispatched = false;
        while step_i < self.schedule.len() || pending_enc.is_some() {
            if self.snapshot_stop.is_some_and(|stop| step_i >= stop) {
                break;
            }
            // Host→Host streaks: skip empty compute encodes/submits between
            // HostOps (was dominating Kitten discrete wall time alongside D2H).
            // A carried encoder is that batch's outstanding device work, so it
            // ends the streak: the batch is finished (submitted) before a host
            // step may read or write the arena.
            let starting_on_host = pending_enc.is_none()
                && step_i < self.schedule.len()
                && self.step_breaks_batch(step_i, &scale, one_op_per_pass, &host_cache, dev);
            if !starting_on_host {
                report.open(Phase::Encode, step_i, &self.schedule);
                let mut enc = match pending_enc.take() {
                    Some(enc) => enc,
                    None => {
                        pass_dispatched = false;
                        dev.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rlx-wgpu run"),
                            })
                    }
                };
                {
                    let mut pass = report.begin_pass(&mut enc);
                    while step_i < self.schedule.len() {
                        if step_is_tail_host(&self.schedule[step_i]) {
                            step_i += 1;
                            continue;
                        }
                        if self.static_once_done && self.static_once_steps.contains(&step_i) {
                            let step = &self.schedule[step_i];
                            if step_runs_on_host(step) {
                                step_i += 1;
                                continue;
                            }
                            if !matches!(step, Step::FftGpu { .. }) {
                                gpu_bi += 1;
                            }
                            step_i += 1;
                            continue;
                        }
                        if step_runs_on_host(&self.schedule[step_i]) {
                            break;
                        }
                        // Vulkan/DX12: end the pass after unary/cast so f32→f16
                        // mirrors are visible to the next step. Only split once
                        // we've dispatched in *this* pass — otherwise the step that
                        // needs the flush would never run (infinite empty passes).
                        if pass_dispatched
                            && step_i > 0
                            && step_needs_pass_flush(
                                &self.schedule[step_i],
                                &self.schedule[step_i - 1],
                            )
                        {
                            break;
                        }
                        // NOTE: the match arms below skip a dispatch whose scaled extent is
                        // zero (`if m_s == 0 { … continue; }` and ~78 siblings). Such a skip
                        // must still consume this step's bind group and advance the cursor —
                        // the same idiom as the `static_once` skip above — because the only
                        // `step_i += 1` for a dispatched step is at the very bottom of this
                        // loop. A bare `continue` re-entered the loop on the same step and
                        // spun forever; a bare `step_i += 1` would shift every later step onto
                        // the wrong bind group.
                        let step = &self.schedule[step_i];
                        // PLAN L3: per-step Perfetto trace span; no-op when
                        // env var RLX_TRACE_PERFETTO unset.
                        let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "wgpu");
                        if rlx_ir::env::var("RLX_DBG_STEP").is_some() {
                            eprintln!("[wgpu-step] {}", step_name(step));
                        }
                        match step {
                            Step::CastF32ToF16 { params } => {
                                // Pre-pass for matmul_coop16: mirror f32 arena
                                // region into f16 shadow buffer so the matmul
                                // kernel can read A as f16. One thread per
                                // element; 64-thread workgroups.
                                if let Some(cast_k) = mm_cast {
                                    pass.set_pipeline(&cast_k.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    let (gx, gy, gz) = dispatch_dims(params.len, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::Matmul {
                                m,
                                n,
                                batch,
                                b_off_f32,
                                b_is_param,
                                compute_precision,
                                ..
                            } =>
                            // The dispatch branches below use a chain of
                            // `is_some() && …unwrap()` to pick a pipeline
                            // because each variant cares about a different
                            // Option<Pipeline>. `if let Some(p) = …` chains
                            // would require nesting per variant; the flat
                            // form is the readable shape here.
                            {
                                #[allow(clippy::unnecessary_unwrap)]
                                // Safe at any batch (see safe_for_active_extent
                                // comment); scale m, output rows past m_s per
                                // batch retain prior values via c_batch_stride.
                                // A schedule carrying the 1×1 pointwise-conv GEMM
                                // never gets an active extent, so `scale` is the
                                // identity there.
                                let m_s = scale(*m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let coop_f16_wide = mm_coop_f16_vk.is_some()
                                    && *compute_precision == MatmulCompute::CoopF16Vk
                                    && crate::coop_f16_vk::use_wide_matmul(
                                        *b_off_f32,
                                        *n,
                                        &self.coop_f16_b_param,
                                        &self.coop_f16_vk_wide_b,
                                    );
                                pass.set_bind_group(
                                    0,
                                    coop_f16_vk_bind_group(self, gpu_bi, coop_f16_wide),
                                    &[],
                                );
                                // Kernel selection priority:
                                //   1. compute_precision == F16 + b_is_param +
                                //      SHADER_F16 → matmul_f16_compute
                                //      (f16 multiply, f32 acc — 2× ALU on Apple)
                                //   2. legacy RLX_WGPU_F16_WEIGHTS opt-in →
                                //      matmul_f16w (storage-only f16; experimental,
                                //      currently regresses on Apple)
                                //   3. wide-N (m≥32, n≥64)   → matmul_wide
                                //   4. otherwise            → matmul (small/skinny)
                                let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
                                if *b_is_param && *compute_precision == MatmulCompute::Bf16Packed {
                                    // Packed-BF16 weight, unpacked in-shader.
                                    // Same 32×32-tile grid as the plain matmul.
                                    crate::backend::bf16_packed_dispatch_inc();
                                    pass.set_pipeline(&mm_bf16w.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if let Some(coop) = mm_coop.as_ref()
                                    && *b_is_param
                                    && *compute_precision == MatmulCompute::Coop16
                                {
                                    // Hardware GEMM via simdgroup_matrix /
                                    // KHR_cooperative_matrix. 32×32 output tile
                                    // per workgroup (16 hardware-GEMM ops with
                                    // shared A/B loads). Caller guaranteed m, n,
                                    // k are multiples of 32/32/8.
                                    pass.set_pipeline(&coop.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if mm_coop_f16_vk.is_some()
                                    && *compute_precision == MatmulCompute::CoopF16Vk
                                {
                                    if coop_f16_wide {
                                        dispatch_wide_f32_matmul(
                                            &mut pass,
                                            mm_w_active,
                                            mm_k,
                                            m_s,
                                            *n,
                                            *batch,
                                        );
                                    } else {
                                        let n_eff = scale(*n);
                                        let coop_vk = matmul_coop_f16_vulkan_active_kernel(
                                            &dev.device,
                                            n_eff,
                                        )
                                        .expect("coop f16 vk kernel missing");
                                        pass.set_pipeline(&coop_vk.pipeline);
                                        pass.dispatch_workgroups(
                                            m_s.div_ceil(16),
                                            n.div_ceil(16),
                                            *batch,
                                        );
                                    }
                                } else if let Some(coop_f32) = mm_coop_f32.as_ref()
                                    && *compute_precision == MatmulCompute::CoopF32
                                {
                                    // CoopF32: Metal uses 32×32 simdgroup tiles;
                                    // Vulkan uses 8×8 coopLoad portable kernel.
                                    // The weight's operand role is deliberately
                                    // not part of this test: the kernel reads
                                    // both operands straight out of the f32
                                    // arena, so the conv GEMM routes (weight in
                                    // A) reach it too. Every family that DOES
                                    // read a weight-side buffer is still kept
                                    // away by the eligibility that admitted it.
                                    coop_f32_dispatch_inc();
                                    pass.set_pipeline(&coop_f32.pipeline);
                                    let (gx, gy) = coop_f32_grid(m_s, *n);
                                    pass.dispatch_workgroups(gx, gy, *batch);
                                } else if let Some(f16c) = mm_f16c.as_ref()
                                    && *b_is_param
                                    && *compute_precision == MatmulCompute::F16
                                {
                                    pass.set_pipeline(&f16c.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if let Some(f16w) = mm_f16w.as_ref()
                                    && *b_is_param
                                    && f16w_opt_in
                                {
                                    pass.set_pipeline(&f16w.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if m_s >= 32 && *n >= 64 {
                                    // Wide path: whole row tiles, then one remainder tile for the rows
                                    // the tile grid would otherwise round up — so a 96-row node computes
                                    // 96 rows, not 128. Both dispatches share this step's bind group, and
                                    // the rule asks for the remainder exactly when the variant ships a
                                    // pipeline to run it (both read `WideVariant::remainder`).
                                    let grid = crate::kernels::wide_grid(
                                        crate::kernels::wide_variant(),
                                        m_s,
                                        *n,
                                    );
                                    pass.set_pipeline(&mm_w_active.pipeline);
                                    pass.dispatch_workgroups(grid.gx, grid.full, *batch);
                                    if grid.tail > 0
                                        && let Some(tail) =
                                            crate::kernels::matmul_wide_tail_kernel(&dev.device)
                                    {
                                        pass.set_pipeline(&tail.pipeline);
                                        pass.dispatch_workgroups(grid.gx, grid.tail, *batch);
                                    }
                                } else {
                                    pass.set_pipeline(&mm_k.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                }
                            }
                            Step::Binary { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_pipeline(&bk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Compare { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Unary { params, f16_mirror } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                if *f16_mirror {
                                    if let Some(uk_f16) = unary_f16_mirror_kernel(&dev.device) {
                                        pass.set_pipeline(&uk_f16.pipeline);
                                    } else {
                                        pass.set_pipeline(&uk.pipeline);
                                    }
                                } else {
                                    pass.set_pipeline(&uk.pipeline);
                                }
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Where { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_pipeline(&wk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Fma { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_pipeline(&fk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ReluBackward { params } | Step::ActivationBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_pipeline(&abk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Reduce { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let rk = reduce_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total_out = outer_s.saturating_mul(params.inner);
                                if params.reduce_dim <= 64 {
                                    // Fast path: 1 thread per output cell.
                                    let (gx, gy, gz) = dispatch_dims(total_out, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    // Tree-reduce path: 1 workgroup (64
                                    // threads) per output cell, parallel
                                    // reduction with shared scratch.
                                    let (gx, gy, gz) = dispatch_dims(total_out, 1);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::Softmax { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = softmax_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → shared-mem
                                // reduction over the softmax axis.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AttentionMask { params } => {
                                // Not active-extent scaled (the step vetoes that
                                // mode for the whole graph), so the rows are the
                                // planned ones — and a step reaching here is one the
                                // route took, which its own floor keeps non-empty.
                                let ak = attention_mask_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One 64-lane workgroup per row: the kernel reads
                                // `wid.x + wid.y * ngs.x` as its row index, which is
                                // the grid convention `softmax_kernel` asserts, so it
                                // is expressed the same way here.
                                let (gx, gy, gz) =
                                    dispatch_dims(params.rows.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                                // One per attention step this route served: the
                                // step is the route's own masked scale.
                                note_attention_decomposed_route();
                            }
                            Step::SoftmaxCrossEntropy { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = softmax_cross_entropy_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SoftmaxCrossEntropyWithLogits { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = softmax_cross_entropy_with_logits_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SoftmaxCrossEntropyBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = softmax_cross_entropy_backward_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::LayerNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let lk = layernorm_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → tree
                                // reduction over the feature dim.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GroupNorm { params } => {
                                if params.groups_total == 0 || params.group_elems == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let gk = group_norm_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per (batch, group)
                                // → tree reduction over the group's elements.
                                let (gx, gy, gz) =
                                    dispatch_dims(params.groups_total.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::LayerNorm2d { params } => {
                                if params.positions == 0 || params.c == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let lk = layer_norm2d_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per (batch, spatial)
                                // position → tree reduction over the channels.
                                // The grid is the POSITION count, not a thread
                                // count — the kernel recovers `pos` from
                                // `workgroup_id` and `num_workgroups` — so the
                                // count is handed over as-is, by asking
                                // `dispatch_dims` for a workgroup size of 1.
                                let (gx, gy, gz) = dispatch_dims(params.positions, 1);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedSwiGLU { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = fused_swiglu_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One thread per output element (outer * n_half).
                                let (gx, gy, gz) =
                                    dispatch_dims(outer_s.saturating_mul(params.n_half), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::RmsNormBackwardInput { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let rk = rms_norm_backward_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::RmsNormBackwardGamma { params }
                            | Step::RmsNormBackwardBeta { params } => {
                                if params.inner == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let rk = rms_norm_backward_param_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::LayerNormBackwardInput { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let lk = layer_norm_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::LayerNormBackwardGammaPartial {
                                params,
                                num_workgroups,
                            } => {
                                if params.inner == 0 || *num_workgroups == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let lk = layer_norm_backward_gamma_partial_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(*num_workgroups, 1, 1);
                            }
                            Step::LayerNormBackwardGammaReduce { params } => {
                                if params.inner == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let lk = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::CumsumBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck = cumsum_backward_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::RopeBackward { params } => {
                                let seq_s = scale(params.seq);
                                if seq_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let rk = rope_backward_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * seq_s * params.hidden;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatherBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let total = outer_s * params.axis_dim * params.trailing;
                                if total > 0 {
                                    let zk = gather_backward_zero_kernel(&dev.device);
                                    pass.set_pipeline(&zk.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    let (gx, _, _) = dispatch_dims(total, 256);
                                    pass.dispatch_workgroups(gx, 1, 1);
                                }
                                let ak = gather_backward_acc_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::Cumsum { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck2 = cumsum_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::CumScan { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck2 = cum_scan_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FftGpu {
                                src_off,
                                dst_off,
                                outer,
                                n,
                                inverse,
                                norm_scale,
                            } => {
                                let res = &self.fft_gpu_steps[fft_i];
                                fft_i += 1;
                                crate::fft_dispatch::dispatch_fft_gpu_in_pass(
                                    &dev.device,
                                    &dev.queue,
                                    &mut pass,
                                    res,
                                    *src_off,
                                    *dst_off,
                                    *outer,
                                    *n,
                                    *inverse != 0,
                                    *norm_scale,
                                );
                            }
                            Step::Copy { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck2 = copy_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Cast { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let cast_k = cast_kernel(&dev.device);
                                pass.set_pipeline(&cast_k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexCast { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = complex_cast_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::BinaryC64 { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = binary_c64_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexNormSq { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = complex_norm_sq_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexNormSqBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = complex_norm_sq_backward_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConjugateC64 { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = conjugate_c64_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FftButterflyStage { params } => {
                                let batch_s = scale(params.batch);
                                let n = batch_s.saturating_mul(params.half);
                                if n == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = fft_butterfly_stage_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::BufferCopy { .. } => {
                                // `step_runs_on_host` classifies a copy as a host
                                // step, so the host-step loop below routes it —
                                // a copy-kernel dispatch (into a batch whose
                                // encoder the join branch carries, or one of its
                                // own) or a host round-trip — and this GPU pass
                                // encodes nothing for it.
                            }
                            Step::ElementwiseRegion { params } => {
                                let len_s = scale(params.len);
                                if len_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                if params.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW
                                {
                                    let ek = elementwise_region_spatial_kernel(&dev.device);
                                    pass.set_pipeline(&ek.pipeline);
                                    let (gx, gy, gz) = dispatch_prologue_nchw(
                                        params.out_w,
                                        params.out_h,
                                        params.out_n * params.out_c,
                                    );
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    let ek = elementwise_region_kernel(&dev.device);
                                    pass.set_pipeline(&ek.pipeline);
                                    let (gx, gy, gz) = dispatch_dims(len_s, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::BatchElementwiseRegion { params } => {
                                let slice_len_s = scale(params.slice_len);
                                let num_batch_s = scale(params.num_batch);
                                if slice_len_s == 0 || num_batch_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ek = batch_elementwise_region_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, _) = dispatch_dims(slice_len_s, 64);
                                pass.dispatch_workgroups(gx, gy, num_batch_s);
                            }
                            Step::Transpose { params, .. } => {
                                // Compute scaled grid count to match the
                                // uniform's scaled out_total when bucket axis
                                // is outermost.
                                let total_s =
                                    if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                        let scaled_d0 = scale(params.out_dim_0);
                                        let inner = params.out_total / params.out_dim_0;
                                        scaled_d0 * inner
                                    } else {
                                        params.out_total
                                    };
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let tk = transpose_kernel(&dev.device);
                                pass.set_pipeline(&tk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Narrow { params } => {
                                let total_s = scale(params.total);
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let nk = narrow_kernel(&dev.device);
                                pass.set_pipeline(&nk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Concat { params } => {
                                let total_s = scale(params.total);
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let cck = concat_kernel(&dev.device);
                                pass.set_pipeline(&cck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Gather { params } => {
                                let n_out_s = scale(params.n_out);
                                if n_out_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let gk = gather_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_out_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatherAxis { params } => {
                                // Trim only the leading (outer) dim; see the
                                // uniform-write path above.
                                let total_s =
                                    scale(params.outer) * params.num_idx * params.trailing;
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let gk = gather_axis_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Attention { params, .. } => {
                                // Scale seq_q for grid dim; per-head strides
                                // come from seq_q_stride / seq_k_stride (full
                                // extent) inside the WGSL.
                                let seq_q_s = scale(params.seq_q);
                                if seq_q_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                // Route from the extents this run walks, not
                                // from the lowered ones: `scale` is the same
                                // active-extent factor the uniform write above
                                // applies to seq_q / seq_k.
                                let decode = attention_decode_eligible(
                                    seq_q_s,
                                    scale(params.seq_k),
                                    params.head_dim,
                                    params.v_head_dim,
                                );
                                let ak = if decode {
                                    // Both kernels are `OnceLock`-cached, so
                                    // this is a load after the first step; like
                                    // the scaled-multiply decode kernels, the
                                    // variant is fetched where it is used rather
                                    // than in the prelude, which would compile
                                    // the shader for graphs that never dispatch
                                    // attention at all.
                                    attention_decode_kernel(&dev.device)
                                } else {
                                    attention_kernel(&dev.device)
                                };
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // Decode form: one workgroup per (batch, head).
                                // Serial form: one thread per (batch, head, row).
                                let (gx, gy, gz) = if decode {
                                    dispatch_dims(params.batch * params.heads, 1)
                                } else {
                                    dispatch_dims(params.batch * params.heads * seq_q_s, 64)
                                };
                                pass.dispatch_workgroups(gx, gy, gz);
                                note_attention_route(decode);
                            }
                            Step::AttentionBackward { params, .. } => {
                                let axis = if params.wrt == 0 {
                                    params.seq_q
                                } else {
                                    params.seq_k
                                };
                                let axis_s = scale(axis);
                                if axis_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ak = attention_bwd_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.heads * axis_s;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Rope { params } => {
                                // Multi-batch via stride-field WGSL fix:
                                // iterate `batch * scaled_seq * last_dim` items.
                                let s_active = scale(params.seq);
                                let total_s = params.batch * s_active * params.last_dim;
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let rk = rope_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::QuantI8 {
                                kernel, threads, ..
                            } => {
                                if *threads == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let k = match kernel {
                                    QuantI8Kernel::Quantize => quantize_i8_kernel(&dev.device),
                                    QuantI8Kernel::Dequantize => dequantize_i8_kernel(&dev.device),
                                };
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(*threads, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::BatchNormInference {
                                kernel, threads, ..
                            } => {
                                if *threads == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let k = match kernel {
                                    BatchNormKernel::Forward => {
                                        batch_norm_inference_kernel(&dev.device)
                                    }
                                    BatchNormKernel::BwdInput => {
                                        batch_norm_inference_bwd_input_kernel(&dev.device)
                                    }
                                    BatchNormKernel::BwdGamma => {
                                        batch_norm_inference_bwd_gamma_kernel(&dev.device)
                                    }
                                    BatchNormKernel::BwdBeta => {
                                        batch_norm_inference_bwd_beta_kernel(&dev.device)
                                    }
                                };
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(*threads, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::QConv2d { threads, .. } => {
                                if *threads == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let k = q_conv2d_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(*threads, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::QMatMul { threads, .. } => {
                                if *threads == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let k = q_matmul_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(*threads, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ScaledLowp {
                                params,
                                kernel,
                                threads,
                            } => {
                                if matches!(kernel, ScaledLowpKernel::MatmulDecode) {
                                    if params.m == 0 || params.n == 0 {
                                        gpu_bi += 1;
                                        step_i += 1;
                                        continue;
                                    }
                                    let k = scaled_matmul_decode_kernel(&dev.device);
                                    pass.set_pipeline(&k.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    // 16x16 output tiles, matching the shader's
                                    // workgroup_size(16, 16).
                                    pass.dispatch_workgroups(
                                        params.n.div_ceil(16),
                                        params.m.div_ceil(16),
                                        1,
                                    );
                                } else {
                                    if *threads == 0 {
                                        gpu_bi += 1;
                                        step_i += 1;
                                        continue;
                                    }
                                    let k = match kernel {
                                        ScaledLowpKernel::QuantScale => {
                                            scaled_quant_scale_kernel(&dev.device)
                                        }
                                        ScaledLowpKernel::Quantize => {
                                            scaled_quantize_kernel(&dev.device)
                                        }
                                        _ => scaled_dequantize_kernel(&dev.device),
                                    };
                                    pass.set_pipeline(&k.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    let (gx, gy, gz) = dispatch_dims(*threads, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::IndexingNd { params, kernel, .. } => {
                                if params.n == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let k = match kernel {
                                    IndexingNdKernel::GatherNd => gather_nd_kernel(&dev.device),
                                    IndexingNdKernel::GatherElements => {
                                        gather_elements_kernel(&dev.device)
                                    }
                                    IndexingNdKernel::ScatterElements => {
                                        scatter_elements_kernel(&dev.device)
                                    }
                                    IndexingNdKernel::ScatterNd => {
                                        scatter_nd_reduce_kernel(&dev.device)
                                    }
                                    IndexingNdKernel::CopySanitize => {
                                        copy_sanitize_kernel(&dev.device)
                                    }
                                };
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(params.n, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Expand { params, .. } => {
                                let total_s =
                                    if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                        let scaled_d0 = scale(params.out_dim_0);
                                        let inner = params.out_total / params.out_dim_0;
                                        scaled_d0 * inner
                                    } else {
                                        params.out_total
                                    };
                                if total_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ek = expand_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Reverse { params, .. } => {
                                let total_s = if params.scale_leading == 1 && params.dim_0 > 0 {
                                    let scaled_d0 = scale(params.dim_0);
                                    let inner = params.total / params.dim_0;
                                    scaled_d0 * inner
                                } else {
                                    params.total
                                };
                                if total_s == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let rk = reverse_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Argmax { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let amk = argmax_kernel(&dev.device);
                                pass.set_pipeline(&amk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool2d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = pool2d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::MaxPool2dBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = maxpool2d_backward_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let gx = params.w.div_ceil(8);
                                let gy = params.h.div_ceil(8);
                                let gz = n_s * params.c;
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::MaxPool3dBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = maxpool3d_backward_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.d * params.h * params.w;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3dBackwardInput { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = conv3d_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c_in * params.d * params.h * params.w;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3dBackwardWeight { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = conv3d_backward_weight_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let c_in_per_g = params.c_in / params.groups.max(1);
                                let total =
                                    params.c_out * c_in_per_g * params.kd * params.kh * params.kw;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GroupNormBackwardInput { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = group_norm_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(n_s * params.num_groups, 1, 1);
                            }
                            Step::GroupNormBackwardGamma { .. } => {
                                let pk = group_norm_backward_gamma_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::GroupNormBackwardBeta { .. } => {
                                let pk = group_norm_backward_beta_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::AxialRope2d { params } => {
                                let batch_s = scale(params.batch);
                                if batch_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let total = batch_s * params.seq * params.hidden;
                                let pk = axial_rope2d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FakeQuantizeFixed { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = fake_quantize_fixed_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FakeQuantizePerBatch { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 || params.chan_dim == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = fake_quantize_perbatch_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(params.chan_dim, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv2d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck2 = conv2d_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // conv2d.wgsl tiles `CONV2D_TILE` output spatial
                                // positions per thread (const must match the kernel).
                                let spatial = params.h_out * params.w_out;
                                let sp_tiles = spatial.div_ceil(CONV2D_TILE);
                                let total = n_s * params.c_out * sp_tiles;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv2dTiled { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck = conv1d_tiled_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // conv1d_tiled.wgsl: one thread per output element
                                // (c_out * l_out), N == 1, w_out == 1.
                                let total = n_s * params.c_out * params.h_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedConvBiasAct { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck = fused_conv_bias_act_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // Same grid as conv2d: TILE (== CONV2D_TILE) output
                                // spatial positions per thread.
                                let spatial = params.h_out * params.w_out;
                                let sp_tiles = spatial.div_ceil(CONV2D_TILE);
                                let total = n_s * params.c_out * sp_tiles;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Im2ColGpu { params } => {
                                // One thread per col element (k_total * spatial);
                                // the following Step::Matmul consumes the col matrix.
                                let imk = im2col2d_kernel(&dev.device);
                                pass.set_pipeline(&imk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.k_total.saturating_mul(params.spatial);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool1d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = pool1d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.l_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let pk = pool3d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv1d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck = conv1d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c_out * params.l_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                // Four output channels per thread when the
                                // convolution is ungrouped; see
                                // conv3d_ctile.wgsl. `RLX_WGPU_CONV3D_SCALAR=1`
                                // forces the one-element-per-thread kernel so
                                // the two can be timed against each other in
                                // one process.
                                let tiled = params.groups <= 1
                                    && !rlx_ir::env::flag("RLX_WGPU_CONV3D_SCALAR");
                                let spatial = params.d_out * params.h_out * params.w_out;
                                let (ck, total) = if tiled {
                                    (
                                        conv3d_ctile_kernel(&dev.device),
                                        n_s * params.c_out.div_ceil(4) * spatial,
                                    )
                                } else {
                                    (conv3d_kernel(&dev.device), n_s * params.c_out * spatial)
                                };
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConvTranspose2d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let ck = conv_transpose2d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c_out * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConvTranspose2dTiled { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let ck = crate::kernels::conv_transpose2d_tiled_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let [gx, gy, gz] = crate::kernels::conv_transpose2d_tiled_grid(
                                    params.w_out,
                                    params.sw,
                                    params.c_out,
                                );
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConvTranspose3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ck = conv_transpose3d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c_out * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ScatterAdd { params } => {
                                let sk = scatter_add_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // Phase 0 zeros the FULL output (preserves
                                // accumulator semantics). Phase 1 scatters first
                                // num_updates_active updates only; serial single
                                // workgroup either way (atomic CAS unsupported in
                                // naga's MSL emitter — see scatter_add.wgsl).
                                if params.op == 0 {
                                    let (gx, gy, gz) = dispatch_dims(params.out_total, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    pass.dispatch_workgroups(1, 1, 1);
                                }
                            }
                            Step::TopK { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let tk = topk_kernel(&dev.device);
                                pass.set_pipeline(&tk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::WelchPeaksGpu { params } => {
                                let batch_s = scale(params.welch_batch);
                                if batch_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let wk = welch_peaks_gpu_kernel(&dev.device);
                                pass.set_pipeline(&wk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(batch_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Frames { params } => {
                                let total = scale(params.total);
                                if total == 0 {
                                    gpu_bi += 1;
                                    step_i += 1;
                                    continue;
                                }
                                let fk = frames_kernel(&dev.device);
                                pass.set_pipeline(&fk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, FRAMES_WORKGROUP);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::UmapKnn { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let uk = umap_knn_kernel(&dev.device);
                                pass.set_pipeline(&uk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::WgpuGpuKernel { name, workgroups } => {
                                // Raw-GPU custom op: fetch the cached pipeline by name
                                // and dispatch against the compile-time bind group.
                                let gk = crate::wgpu_gpu_custom::lookup(name).expect(
                                "WgpuGpuKernel vanished from the registry between compile and run",
                            );
                                let kernel = crate::wgpu_gpu_custom::get_or_build_pipeline(
                                    &dev.device,
                                    &*gk,
                                );
                                pass.set_pipeline(&kernel.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = *workgroups;
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GroupedMatmul { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let gk = grouped_matmul_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                            }
                            Step::ScaledGroupedMatmul { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = scaled_grouped_matmul_decode_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                            }
                            Step::Sample { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let sk = sample_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SelectiveScan { params } => {
                                // Predicate-gated to batch=1; the seq scaling
                                // happens inside the kernel (uniform sees scaled
                                // seq). Dispatch grid here is per-(batch, hidden);
                                // unaffected by seq scaling.
                                let ssk = selective_scan_kernel(&dev.device);
                                pass.set_pipeline(&ssk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.hidden;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatedDeltaNet {
                                params,
                                use_gpu: true,
                                ..
                            } => {
                                // One workgroup per (batch, head); workgroup_size=128.
                                let gk = gated_delta_net_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch * params.heads, 1, 1);
                            }
                            Step::Mamba2 { params } => {
                                let mk = mamba2_kernel(&dev.device);
                                pass.set_pipeline(&mk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.heads * params.head_dim;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Gru { params } => {
                                // One workgroup per batch item (workgroup_size=256).
                                let gk = gru_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch, 1, 1);
                            }
                            Step::Rnn { params } => {
                                let rk = rnn_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch, 1, 1);
                            }
                            Step::LstmGpu { params } => {
                                // One workgroup per batch item (workgroup_size=256).
                                let lk = lstm_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch, 1, 1);
                            }
                            Step::DequantMatmul { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let dk = dequant_matmul_kernel(&dev.device);
                                pass.set_pipeline(&dk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                            }
                            Step::DequantMatmulMlx { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let dk = dequant_matmul_mlx_kernel(&dev.device);
                                pass.set_pipeline(&dk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup per (col, row_tile); local size 256
                                // splits K and stages X in workgroup memory.
                                let n_row_tiles = m_s.div_ceil(8);
                                pass.dispatch_workgroups(params.n * n_row_tiles, 1, 1);
                            }
                            Step::FusedResidualLn { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let frk = fused_residual_ln_kernel(&dev.device);
                                pass.set_pipeline(&frk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → tree
                                // reduction over the feature dim.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedResidualLnTee { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let frtk = fused_residual_ln_tee_kernel(&dev.device);
                                pass.set_pipeline(&frtk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → tree
                                // reduction over the feature dim.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedResidualRmsNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let frk = fused_residual_rms_norm_kernel(&dev.device);
                                pass.set_pipeline(&frk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → tree
                                // reduction over the feature dim.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AdaLayerNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ak = ada_layer_norm_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup (64 threads) per row → tree
                                // reduction over the feature dim.
                                let (gx, gy, gz) = dispatch_dims(outer_s.saturating_mul(64), 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatedResidual { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let total = outer_s.saturating_mul(params.inner);
                                let gk = gated_residual_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AdaLayerNormBackward { params } => {
                                let mod_rows_s = scale(params.mod_rows);
                                if mod_rows_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let ak = ada_layer_norm_backward_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(mod_rows_s, 1, 1);
                            }
                            Step::GatedResidualBackward { params } => {
                                let mod_rows_s = scale(params.mod_rows);
                                if mod_rows_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let gk = gated_residual_backward_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(mod_rows_s, 1, 1);
                            }
                            Step::MatmulQkv { params, kind } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    if !matches!(step, Step::FftGpu { .. }) {
                                        gpu_bi += 1;
                                    }
                                    step_i += 1;
                                    continue;
                                }
                                let qkv_coop_wide = matches!(kind, MatmulQkvKind::CoopF16Vk)
                                    && crate::coop_f16_vk::use_wide_matmul(
                                        params.b_off,
                                        params.n,
                                        &self.coop_f16_b_param,
                                        &self.coop_f16_vk_wide_b,
                                    );
                                pass.set_bind_group(
                                    0,
                                    coop_f16_vk_bind_group(self, gpu_bi, qkv_coop_wide),
                                    &[],
                                );
                                match kind {
                                    MatmulQkvKind::CoopF16Vk => {
                                        if qkv_coop_wide {
                                            pass.set_pipeline(
                                                &matmul_qkv_kernel(&dev.device).pipeline,
                                            );
                                            pass.dispatch_workgroups(
                                                params.n.div_ceil(32),
                                                m_s.div_ceil(32),
                                                1,
                                            );
                                        } else {
                                            let n_eff = scale(params.n);
                                            let mqk = matmul_qkv_coop_f16_vk_active_kernel(
                                                &dev.device,
                                                n_eff,
                                            )
                                            .expect("coop f16 matmul_qkv kernel missing");
                                            pass.set_pipeline(&mqk.pipeline);
                                            pass.dispatch_workgroups(
                                                m_s.div_ceil(16),
                                                params.n.div_ceil(16),
                                                1,
                                            );
                                        }
                                    }
                                    MatmulQkvKind::CoopF32 => {
                                        pass.set_pipeline(
                                            &matmul_qkv_coop_f32_kernel(&dev.device)
                                                .expect("coop matmul_qkv kernel missing")
                                                .pipeline,
                                        );
                                        pass.dispatch_workgroups(
                                            params.n.div_ceil(32),
                                            m_s.div_ceil(32),
                                            1,
                                        );
                                    }
                                    MatmulQkvKind::F32 => {
                                        pass.set_pipeline(&matmul_qkv_kernel(&dev.device).pipeline);
                                        pass.dispatch_workgroups(
                                            params.n.div_ceil(32),
                                            m_s.div_ceil(32),
                                            1,
                                        );
                                    }
                                }
                            }
                            Step::GatherSplit { .. }
                            | Step::DequantMatmulGguf { .. }
                            | Step::DequantMatmulInt8Host { .. }
                            | Step::DequantMatmulMlxHost { .. }
                            | Step::Conv2dHost { .. }
                            | Step::DequantGroupedMatmulGguf { .. }
                            | Step::DequantGroupedMatmulMlxHost { .. }
                            | Step::GatedDeltaNet { use_gpu: false, .. }
                            | Step::Lstm { .. }
                            | Step::ConvTranspose2dHost { .. }
                            | Step::ConvTranspose3dHost { .. }
                            | Step::GroupNormHost { .. }
                            | Step::LayerNorm2dHost { .. }
                            | Step::ResizeNearest2xHost { .. }
                            | Step::ReverseHost { .. }
                            | Step::ArgReduceHost { .. }
                            | Step::AxialRope2dHost { .. }
                            | Step::GruHost { .. }
                            | Step::RnnHost { .. }
                            | Step::Llada2GroupLimitedGate { .. }
                            | Step::UmapKnnHost { .. }
                            | Step::MsDeformAttnHost { .. }
                            | Step::CollectiveHost { .. }
                            | Step::CustomHost { .. }
                            | Step::FftHost { .. }
                            | Step::FftQHost { .. }
                            | Step::ScanHost { .. }
                            | Step::HostOp { .. }
                            | Step::CpuIndexing { .. }
                            | Step::ConcatHost { .. }
                            | Step::ConcatHostPieces { .. }
                            | Step::TransposeHost { .. }
                            | Step::NarrowHost { .. }
                            | Step::ExpandHost { .. }
                            | Step::SpdHost { .. }
                            | Step::Im2ColHost { .. }
                            | Step::Conv2dBackwardWeightHost { .. }
                            | Step::Conv2dBackwardInputHost { .. }
                            | Step::RngNormalHost { .. }
                            | Step::RngUniformHost { .. }
                            | Step::WelchPeaksHost { .. }
                            | Step::LogMelHost { .. }
                            | Step::LogMelBackwardHost { .. } => {}
                            #[cfg(feature = "splat")]
                            Step::GaussianSplatRender { .. }
                            | Step::GaussianSplatRenderBackward { .. }
                            | Step::GaussianSplatPrepare { .. }
                            | Step::GaussianSplatRasterize { .. } => {}
                        }
                        if !matches!(step, Step::FftGpu { .. }) {
                            gpu_bi += 1;
                        }
                        if rlx_ir::env::flag("RLX_WGPU_DBG_COUNTERS") {
                            eprintln!(
                                "[wgpu-bi] step {step_i} {} -> gpu_bi now {gpu_bi}",
                                step_name(step)
                            );
                        }
                        step_i += 1;
                        pass_dispatched = true;
                        if one_op_per_pass {
                            break; // end the pass after this op ⇒ barrier before next
                        }
                    }
                }
                // A joined packed multiply: end the pass, encode its dispatches
                // into THIS encoder, and carry the batch on. Its scratch stays alive
                // because the encode is contiguous inside this one encoder — dispatch,
                // then `copy_buffer_to_buffer` into the arena — and a later call that
                // reuses the pooled out slab can only do so after that copy in command
                // order; the slab's buffer is not released while a submitted-or-carried
                // command buffer references it. It consumes no lane either: lowering
                // pushed none for this step and the run-time cursors skip
                // host-classified steps, so no later device step's slot moves.
                if !one_op_per_pass
                    && let Some(step) = self.schedule.get(step_i)
                    && self.joined_dequant_gguf(step, &scale)
                    && let Step::DequantMatmulGguf {
                        m,
                        k,
                        n,
                        scheme_id,
                        x_byte_off,
                        w_byte_off,
                        out_byte_off,
                    } = step
                {
                    // This call reads the arena on the device, so a host result still
                    // only in the mirror has to reach the device before this batch is
                    // submitted. Reached at a batch's start (a host step never runs
                    // while an encoder is carried), so the write lands with the
                    // submission that carries the joined dispatches.
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                        crate::backend::note_device_wait();
                        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    }
                    // The admission is the ladder's device-only answers, and the call
                    // can still be refused here: the encode's gates read arena, adapter
                    // and lever facts a device-free ladder cannot see (see
                    // [`crate::gguf_gpu::DequantMatmulPath::GroupedDevice`]). The refusal
                    // is not run past — an unencoded call leaves `step_i` where it is,
                    // so this iteration's tail submits the batch and the outer host
                    // dispatch then runs its own arm on an arena that submission has made
                    // coherent, handing the refusal to the entry point that owns its
                    // recompute. The mirror this batch may have been carrying was flushed
                    // by the block above, the same door a plain device step goes through.
                    let encoded = crate::gguf_gpu::encode_dequant_matmul_gguf_into(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        &mut enc,
                        scale(*m) as usize,
                        *k as usize,
                        *n as usize,
                        *scheme_id,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *out_byte_off as usize,
                        &mut report.marks(),
                    );
                    if encoded {
                        step_i += 1;
                        pass_dispatched = true;
                        pending_enc = Some(enc);
                        continue;
                    }
                }
                // A joined copy: the same shape as the packed multiply above — end
                // the pass, encode the copy's work into THIS encoder, carry the
                // batch on. The route value comes from `buffer_copy_route_now`; its
                // doc has why this branch and the batch boundary cannot disagree.
                //
                // The dispatch's own pass, and the reason nothing is staged, are
                // `encode_arena_copy_dispatch`'s doc. The mirror's entry for `dst` is
                // invalidated here instead of after a submission — the same point in
                // the work's life, since no host step can run while an encoder is
                // carried and the route refuses to join while any deferred write is
                // pending. `pass_dispatched` keeps the whole-mirror clear at the next
                // host boundary firing.
                if !one_op_per_pass
                    && let Some(step) = self.schedule.get(step_i)
                    && let Some((base, size)) = self
                        .buffer_copy_route_now(step, &host_cache, dev)
                        .and_then(CopyRoute::join_window)
                    && let Step::BufferCopy {
                        src_byte_off,
                        dst_byte_off,
                        bytes,
                    } = step
                {
                    self.encode_arena_copy_dispatch(
                        dev,
                        &mut enc,
                        *src_byte_off,
                        *dst_byte_off,
                        *bytes as u64,
                        base,
                        size,
                    );
                    host_cache.invalidate(*dst_byte_off as usize);
                    step_i += 1;
                    pass_dispatched = true;
                    pending_enc = Some(enc);
                    continue;
                }
                // `step_runs_on_host` on purpose, not `step_breaks_batch`: the two
                // agree here. A call the join branch consumed never reaches this site;
                // one whose encode refused it does, and both predicates answer false
                // for it — its own step is host-classified, and `step_needs_pass_flush`
                // has no arm for a tail-host step kind and is false for every
                // `DequantMatmulGguf` arm, so no drain is asked for here either way. A
                // *joined copy* can be the `prev` step this reads: on a Vulkan/DX12-class
                // adapter that arm then asks for the drain, which stands in for the
                // boundary the copy's own submission used to provide (see
                // `step_breaks_batch`'s doc).
                let needs_f16_drain = step_i < self.schedule.len()
                    && !step_runs_on_host(&self.schedule[step_i])
                    && step_i > 0
                    && step_needs_pass_flush(&self.schedule[step_i], &self.schedule[step_i - 1]);
                let gpu_schedule_done = step_i >= self.schedule.len();
                let skip_readback =
                    rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") || self.dispatch_only;
                let defer_tail = gpu_schedule_done && self.schedule.iter().any(step_is_tail_host);
                let mut fused_readback: Option<(
                    ReadbackLayout,
                    std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
                    Vec<usize>,
                )> = None;
                if gpu_schedule_done && !skip_readback && !defer_tail {
                    if !self.gpu_handle_feeds.is_empty() {
                        self.propagate_gpu_handle_feeds_on_gpu(dev, &mut enc);
                    }
                    let plan = self.readback_plan();
                    let out_ids_all: Vec<_> = self.graph.outputs.clone();
                    let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                    let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                    if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                        if self.tiny_readback.is_none() {
                            self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                        }
                        let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                        encode_readback_copies(
                            &mut enc,
                            &self.arena,
                            tiny.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut enc, tiny.buffer(), &layout);
                        crate::backend::note_device_submit();
                        let sub = dev.queue.submit(std::iter::once(enc.finish()));
                        report.open(Phase::Wait, step_i, &self.schedule);
                        wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                        map_rx.recv().unwrap().unwrap();
                        report.open(Phase::Readback, step_i, &self.schedule);
                        self.debug_scans_after_run(dev);
                        // Decode before arming: `tiny` borrows `self` immutably and the
                        // borrow must end before `arm_static_once` takes it mutably.
                        let decoded =
                            vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)];
                        self.arm_static_once();
                        return self.pack_readback_outputs(&plan, decoded);
                    }
                    ReadbackStaging::prepare(
                        &dev.device,
                        &mut self.readback_staging,
                        layout.total_bytes,
                    );
                    if let Some(staging) = self.readback_staging.as_ref() {
                        encode_readback_copies(
                            &mut enc,
                            &self.arena,
                            staging.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut enc, staging.buffer(), &layout);
                        fused_readback = Some((layout, map_rx, plan));
                    }
                }
                crate::backend::note_device_submit();
                let main_submission = dev.queue.submit(std::iter::once(enc.finish()));
                if defer_tail {
                    report.open(Phase::Wait, step_i, &self.schedule);
                    crate::backend::note_device_wait();
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    report.open(Phase::Host, step_i, &self.schedule);
                    self.run_tail_host_audio_ops(dev);
                    if !skip_readback {
                        report.open(Phase::Readback, step_i, &self.schedule);
                        let mut rb_enc =
                            dev.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rlx-wgpu readback after tail-host"),
                                });
                        if !self.gpu_handle_feeds.is_empty() {
                            self.propagate_gpu_handle_feeds_on_gpu(dev, &mut rb_enc);
                        }
                        let plan = self.readback_plan();
                        let out_ids_all: Vec<_> = self.graph.outputs.clone();
                        let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                        let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                        if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                            if self.tiny_readback.is_none() {
                                self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                            }
                            let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                            encode_readback_copies(
                                &mut rb_enc,
                                &self.arena,
                                tiny.buffer(),
                                &out_ids,
                                &layout,
                            );
                            let map_rx = schedule_readback_map(&mut rb_enc, tiny.buffer(), &layout);
                            crate::backend::note_device_submit();
                            let sub = dev.queue.submit(std::iter::once(rb_enc.finish()));
                            wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                            map_rx.recv().unwrap().unwrap();
                            self.debug_scans_after_run(dev);
                            // Decode before arming: `tiny` borrows `self` immutably and the
                            // borrow must end before `arm_static_once` takes it mutably.
                            let decoded =
                                vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)];
                            self.arm_static_once();
                            return self.pack_readback_outputs(&plan, decoded);
                        }
                        ReadbackStaging::prepare(
                            &dev.device,
                            &mut self.readback_staging,
                            layout.total_bytes,
                        );
                        if let Some(staging) = self.readback_staging.as_ref() {
                            encode_readback_copies(
                                &mut rb_enc,
                                &self.arena,
                                staging.buffer(),
                                &out_ids,
                                &layout,
                            );
                            let map_rx =
                                schedule_readback_map(&mut rb_enc, staging.buffer(), &layout);
                            crate::backend::note_device_submit();
                            let sub = dev.queue.submit(std::iter::once(rb_enc.finish()));
                            wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                            map_rx.recv().unwrap().unwrap();
                            self.debug_scans_after_run(dev);
                            let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                            self.arm_static_once();
                            return self.pack_readback_outputs(&plan, partial);
                        }
                    }
                }
                if needs_f16_drain {
                    report.open(Phase::Wait, step_i, &self.schedule);
                    crate::backend::note_device_wait();
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                }
                // A joined call — a packed multiply or a device-route copy — is
                // host-classified but is not a host sync here; `step_breaks_batch`'s
                // doc has the site list. This is also the one place where the
                // widened `step_is_tail_host` term is not equivalent to
                // `step_runs_on_host(next)`: under `RLX_WGPU_ONE_OP_PER_PASS` the
                // pass loop's one-op `break` can leave `step_i` on a tail-host step,
                // and this then adds a poll and a mirror clear the old predicate did
                // not ask for — harmless in that debugging mode (it only drains work
                // already submitted), and recorded so the claim is exact.
                let need_host_sync = step_i < self.schedule.len()
                    && self.step_breaks_batch(step_i, &scale, one_op_per_pass, &host_cache, dev);
                if need_host_sync {
                    report.open(Phase::Wait, step_i, &self.schedule);
                    crate::backend::note_device_wait();
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    // Only invalidate after real GPU work. Host→Host chains share
                    // one outer loop iteration's empty encode; clearing there made
                    // HostTensorCache always miss (full D2H every HostOp). This is
                    // also where a joined batch's write invalidates the mirror: its
                    // carried encoder is submitted by the iteration after the join,
                    // which reaches this site with `pass_dispatched` set and the next
                    // step a host step (the schedule's end leaves `need_host_sync`
                    // false, and there the final flush below drains the device).
                    if pass_dispatched {
                        host_cache.clear();
                    }
                }
                if gpu_schedule_done {
                    // The schedule's end can be reached from inside the batch — the
                    // paths below return or break here — so it is read at this edge
                    // too, not only at the loop's own exit.
                    report.end_of_schedule(step_i, &self.schedule);
                    if skip_readback || defer_tail {
                        self.arm_static_once();
                        return self
                            .graph
                            .outputs
                            .iter()
                            .map(|&id| {
                                let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                                vec![0.0; n]
                            })
                            .collect();
                    }
                    if let (Some((layout, map_rx, plan)), Some(staging)) =
                        (fused_readback, self.readback_staging.as_ref())
                    {
                        report.open(Phase::Wait, step_i, &self.schedule);
                        wait_readback_map(
                            &dev.device,
                            main_submission,
                            &map_rx,
                            layout.total_bytes,
                        );
                        map_rx.recv().unwrap().unwrap();
                        report.open(Phase::Readback, step_i, &self.schedule);
                        self.debug_scans_after_run(dev);
                        let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                        self.arm_static_once();
                        return self.pack_readback_outputs(&plan, partial);
                    }
                    break;
                }
            } // !starting_on_host — GPU encode/submit/readback path

            // Skip already-baked static weight packing (Concat/Expand of Params).
            while step_i < self.schedule.len()
                && self.static_once_done
                && self.static_once_steps.contains(&step_i)
                && step_runs_on_host(&self.schedule[step_i])
            {
                step_i += 1;
            }
            if step_i >= self.schedule.len() {
                break;
            }
            if !step_runs_on_host(&self.schedule[step_i]) {
                continue;
            }
            report.open(Phase::Host, step_i, &self.schedule);
            // Does this host step go straight at the device, bypassing the
            // mirror? The cache-aware steps below read and write through
            // `host_cache` themselves; everything else (the `with_whole_arena`
            // family — GroupNorm, LayerNorm2d, Reverse, Gru, …) reads the
            // device and rewrites the arena underneath it.
            //
            // Such a step needs the mirror flushed BEFORE it runs (so it sees
            // pending host results) *and* dropped AFTER (so the next cache-aware
            // step does not serve a pre-step copy from the mirror). Only the
            // first half used to happen: a `HostOp` following a hosted
            // GroupNorm re-read its stale cached input, silently, and nothing
            // cleared the mirror because `host_cache.clear()` is reached only
            // after real GPU work (`pass_dispatched`) — a run of consecutive
            // host steps never gets there.
            let bypasses_host_cache = !matches!(
                &self.schedule[step_i],
                Step::HostOp { .. }
                    | Step::Conv2dHost { .. }
                    | Step::ExpandHost { .. }
                    | Step::NarrowHost { .. }
                    | Step::TransposeHost { .. }
                    | Step::ConcatHost { .. }
                    | Step::BufferCopy { .. }
            );
            if host_cache.has_deferred_writes() && bypasses_host_cache {
                let mut a = crate::host_stage::WgpuArena {
                    arena: &self.arena,
                    device: &dev.device,
                    queue: &dev.queue,
                    size_bytes: 0,
                };
                host_cache.flush_to_device(&mut a);
                crate::backend::note_device_wait();
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
            match &self.schedule[step_i] {
                Step::BufferCopy {
                    src_byte_off,
                    dst_byte_off,
                    bytes,
                } => {
                    let src = *src_byte_off;
                    let dst = *dst_byte_off;
                    let nbytes = *bytes as u64;
                    let src_is_weight = crate::buffer::is_weight_off(src as usize);
                    // The route this arm runs by — `buffer_copy_route_now`'s doc has
                    // why it is the same answer the batch boundary joined on.
                    let route = self
                        .buffer_copy_route_now(&self.schedule[step_i], &host_cache, dev)
                        .expect("this arm is a BufferCopy");
                    if route == CopyRoute::Host {
                        let n_f32 = (nbytes as usize) / 4;
                        let src_u = src as usize;
                        let dst_u = dst as usize;
                        let data = if let Some(hit) = host_cache.get_arc_covering(src_u, n_f32) {
                            hit[..n_f32].to_vec()
                        } else {
                            if host_cache.is_dirty(src_u) {
                                let mut a = crate::host_stage::WgpuArena {
                                    arena: &self.arena,
                                    device: &dev.device,
                                    queue: &dev.queue,
                                    size_bytes: 0,
                                };
                                host_cache.flush_offset(&mut a, src_u);
                            }
                            let bytes_host = self.arena.read_bytes_range(
                                &dev.device,
                                &dev.queue,
                                src_u,
                                nbytes as usize,
                            );
                            bytemuck::cast_slice(&bytes_host).to_vec()
                        };
                        // A weight-tagged source is a *staged* copy, and stages
                        // are bump-allocated into a wrapping scratch reserve —
                        // so many params share one `dst`. The deferred host
                        // mirror is keyed by `dst`, so deferring these collapses
                        // the whole sequence to whichever copy ran last and
                        // every other weight reads stale bytes. Silently: the
                        // graph stays finite and simply stops depending on its
                        // input. Only `RLX_WGPU_HOST_EAGER_H2D` used to prevent
                        // it, which made the flag a correctness switch rather
                        // than the performance knob it is meant to be.
                        let defer = !src_is_weight && !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");
                        if !defer {
                            self.arena.write_bytes_range(
                                &dev.queue,
                                dst_u,
                                bytemuck::cast_slice(data.as_slice()),
                            );
                            crate::backend::note_device_wait();
                            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                        }
                        host_cache.insert(dst_u, data, defer);
                        step_i += 1;
                        continue;
                    }
                    // `CopyRoute::Skip` falls through to the per-step
                    // bookkeeping below.
                    if let CopyRoute::Device { base, size } = route {
                        // Dead code: `deferred_writes` above already routed
                        // every copy with a pending mirror to the host, so
                        // this arm cannot see one. Kept so the flush is
                        // already correct if that condition ever changes.
                        if host_cache.has_deferred_writes() {
                            let mut a = crate::host_stage::WgpuArena {
                                arena: &self.arena,
                                device: &dev.device,
                                queue: &dev.queue,
                                size_bytes: 0,
                            };
                            host_cache.flush_to_device(&mut a);
                            crate::backend::note_device_wait();
                            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                        }
                        // The copy's own submission: reached when this run did not
                        // join the call — the step-by-step and snapshot modes, which
                        // keep one operation per submission on purpose. Anything else
                        // reaching here is a call the join branch declined, so the
                        // arm stays self-contained rather than assume it cannot.
                        let mut enc =
                            dev.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rlx-wgpu arena_copy"),
                                });
                        self.encode_arena_copy_dispatch(
                            dev, &mut enc, src, dst, nbytes, base, size,
                        );
                        crate::backend::note_device_submit();
                        dev.queue.submit(std::iter::once(enc.finish()));
                        crate::backend::note_device_wait();
                        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                        host_cache.invalidate(dst as usize);
                    }
                }
                Step::GatherSplit {
                    n_out,
                    n_idx,
                    dim,
                    vocab,
                    table_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    run_gather_split(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *n_out,
                        *n_idx,
                        *dim,
                        *vocab,
                        *table_byte_off as usize,
                        *idx_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::DequantMatmulGguf {
                    m,
                    k,
                    n,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    // Active-extent: scale m (seq·batch) like Metal/other matmuls.
                    let mm = scale(*m) as usize;
                    let kk = *k as usize;
                    let nn = *n as usize;
                    // The ladder's own selection — the same call the batch
                    // boundary joins on (`joined_dequant_gguf`), so a call this arm
                    // runs unjoined is one the boundary deliberately left alone (the
                    // step-by-step / snapshot modes, or a route outside the join set)
                    // rather than a second opinion about the route.
                    // - The tiled one-row arm with an addressable X window — the two
                    //   the batch boundary joins:
                    //   [`crate::gguf_gpu::DequantMatmulPath::OneRowDevice`] at one
                    //   row and [`crate::gguf_gpu::DequantMatmulPath::GroupedDevice`]
                    //   above it. gemv_rows re-checks the same window guard, so
                    //   neither reaches the host. One row is the per-row loop; above
                    //   one row it is the row-grouped GEMM (one pass over each packed
                    //   weight tile per group of rows; a split Q4_K chunk reads each
                    //   block's 16-byte header twice), whose arm N-chunks the weight
                    //   window itself, so a weight parked in the weight buffer or
                    //   wider than the binding limit still takes it — and the per-row
                    //   loop is that arm's own fallback.
                    // - The fused arm: Q1_0 at every row (its tiled GEMM above one row
                    //   while `RLX_WGPU_Q1_0_GEMM_DISABLE` is off and the windows fit
                    //   a shard, else the per-row loop), and a call of the tiled arm
                    //   whose X window the guard refuses — gemv_rows' own check of
                    //   that guard hands it to gguf_host (`Host`), never to a device
                    //   arm.
                    // - The scratch arm: an in-arena f32 slab then `matmul_bt` (`gemv`
                    //   schemes are handed back to the per-row loop above, so a slab
                    //   can still record PerRow), and the host route that entry point
                    //   short-circuits a sharded arena or a parked weight to. Its
                    //   grouped arm applies the very gates the ladder's admission did,
                    //   so a call the ladder kept out of `GroupedDevice` that way lands
                    //   on the slab and not on that arm. A scheme whose flat pack
                    //   is block-linear over n·k needs k a whole number of blocks to
                    //   slice rows; unaligned k there used to force the host D2H/H2D
                    //   fallback for *all* m>1.
                    //
                    // The device answers honour the encode's answer: a call the fused
                    // entry point cannot encode — Q8_0's grouped arm, a multi-row call
                    // its ladder's per-row loop does not carry —
                    // falls to the scratch arm's entry point, which owns its recompute.
                    // So a call the boundary joined reaches this arm exactly when its
                    // encode refused it, or when the run does not batch at all
                    // (one-op-per-pass, the snapshot).
                    let path = crate::gguf_gpu::select_path(
                        mm,
                        kk,
                        *scheme_id,
                        *x_byte_off as usize,
                        self.dequant_scratch_off,
                    );
                    if path == crate::gguf_gpu::DequantMatmulPath::Host {
                        crate::gguf_gpu::note_route(crate::gguf_gpu::DequantMatmulRoute::Host);
                        crate::gguf_host::run_dequant_matmul_gguf(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            mm,
                            kk,
                            nn,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    } else {
                        // A scratch answer names the scratch entry point, so it is not
                        // tried in the fused one: that entry point owns the route for
                        // every shape this ladder sends to it, and its `is_sharded` /
                        // `is_weight_off` short-circuits are the reason it must see them
                        // — they stand in front of the per-row loop it delegates to. A
                        // scratch answer is any call the gates above keep out of the two
                        // device arms with a slab booked: a single row of a scheme
                        // neither the fused entry point nor the tiled one-row arm serves,
                        // a contraction that is not a whole number of the scheme's own
                        // packing blocks, a multi-row call of a scheme neither arm
                        // serves, and an X window the grouped GEMM's `vec4` reads
                        // cannot address.
                        // Everything else is tried in the fused entry point, and one it
                        // does not run falls to the same recompute below.
                        let ran = path != crate::gguf_gpu::DequantMatmulPath::ScratchArm
                            && crate::gguf_gpu::run_dequant_matmul_gguf_gemv_rows(
                                &self.arena,
                                &dev.device,
                                &dev.queue,
                                mm,
                                kk,
                                nn,
                                *scheme_id,
                                *x_byte_off as usize,
                                *w_byte_off as usize,
                                *out_byte_off as usize,
                                &mut report.marks(),
                            );
                        if !ran {
                            crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                                &self.arena,
                                &dev.device,
                                &dev.queue,
                                mm,
                                kk,
                                nn,
                                *scheme_id,
                                *x_byte_off as usize,
                                *w_byte_off as usize,
                                self.dequant_scratch_off,
                                *out_byte_off as usize,
                                &mut report.marks(),
                            );
                        }
                    }
                }
                Step::DequantMatmulInt8Host {
                    m,
                    k,
                    n,
                    block_size,
                    is_asymmetric,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    out_byte_off,
                } => {
                    let mm = scale(*m) as usize;
                    crate::int8_host::run_dequant_matmul_int8(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        mm,
                        *k as usize,
                        *n as usize,
                        *block_size as usize,
                        *is_asymmetric,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::DequantMatmulMlxHost {
                    m,
                    k,
                    n,
                    scheme,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    out_byte_off,
                } => {
                    let mm = scale(*m) as usize;
                    crate::gguf_host::run_dequant_matmul_mlx(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        mm,
                        *k as usize,
                        *n as usize,
                        *scheme,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::Conv2dHost {
                    n,
                    c_in,
                    c_out,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    in_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    crate::conv_host::run_conv2d_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *n as usize,
                        *c_in as usize,
                        *c_out as usize,
                        *h as usize,
                        *w as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kh as usize,
                        *kw as usize,
                        *sh as usize,
                        *sw as usize,
                        *ph as usize,
                        *pw as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                        *in_byte_off as usize,
                        *w_byte_off as usize,
                        *out_byte_off as usize,
                        Some(&mut host_cache),
                    );
                }
                Step::DequantGroupedMatmulGguf {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    if self.dequant_scratch_off > 0 {
                        crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                            &mut report.marks(),
                        );
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
                }
                Step::DequantGroupedMatmulMlxHost {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    idx_byte_off,
                    out_byte_off,
                    scale_bf16,
                } => {
                    crate::gguf_host::run_dequant_grouped_matmul_mlx(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        scale(*m) as usize,
                        *k as usize,
                        *n as usize,
                        *num_experts as usize,
                        *scheme,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *idx_byte_off as usize,
                        *out_byte_off as usize,
                        *scale_bf16,
                    );
                }
                Step::GatedDeltaNet {
                    params,
                    q_byte_off,
                    k_byte_off,
                    v_byte_off,
                    g_byte_off,
                    beta_byte_off,
                    state_byte_off,
                    dst_byte_off,
                    use_gpu: false,
                } => {
                    crate::gdn_host::run_gated_delta_net(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *q_byte_off as usize,
                        *k_byte_off as usize,
                        *v_byte_off as usize,
                        *g_byte_off as usize,
                        *beta_byte_off as usize,
                        *state_byte_off as usize,
                        *dst_byte_off as usize,
                        params.batch as usize,
                        scale(params.seq) as usize,
                        params.heads as usize,
                        params.state_size as usize,
                        params.use_carry != 0,
                        params.gate_per_channel != 0,
                    );
                }
                Step::GatedDeltaNet { use_gpu: true, .. } => {
                    unreachable!("rlx-wgpu: GPU GatedDeltaNet handled in compute pass");
                }
                Step::Lstm {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    bias_byte_off,
                    h0_byte_off,
                    c0_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    crate::lstm_host::run_lstm(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *bias_byte_off as usize,
                        *h0_byte_off as usize,
                        *c0_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                    );
                }
                Step::ConvTranspose2dHost {
                    src_byte_off,
                    weight_byte_off,
                    dst_byte_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                } => {
                    crate::conv_transpose2d_host::run_conv_transpose2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *weight_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c_in as usize,
                        *h as usize,
                        *w_in as usize,
                        *c_out as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kh as usize,
                        *kw as usize,
                        *sh as usize,
                        *sw as usize,
                        *ph as usize,
                        *pw as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                    );
                }
                Step::ConvTranspose3dHost {
                    src_byte_off,
                    weight_byte_off,
                    dst_byte_off,
                    n,
                    c_in,
                    d,
                    h,
                    w_in,
                    c_out,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    groups,
                } => {
                    crate::conv_transpose3d_host::run_conv_transpose3d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *weight_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c_in as usize,
                        *d as usize,
                        *h as usize,
                        *w_in as usize,
                        *c_out as usize,
                        *d_out as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kd as usize,
                        *kh as usize,
                        *kw as usize,
                        *sd as usize,
                        *sh as usize,
                        *sw as usize,
                        *pd as usize,
                        *ph as usize,
                        *pw as usize,
                        *dd as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                    );
                }
                Step::GroupNormHost {
                    src_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps,
                } => {
                    crate::vision_host::run_group_norm(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                        *num_groups as usize,
                        *eps,
                    );
                }
                Step::LayerNorm2dHost {
                    src_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                    eps,
                } => {
                    crate::vision_host::run_layer_norm2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                        *eps,
                    );
                }
                Step::ResizeNearest2xHost {
                    src_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    crate::vision_host::run_resize_nearest_2x(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                    );
                }
                Step::ReverseHost {
                    src_byte_off,
                    dst_byte_off,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    crate::vision_host::run_reverse(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        dims,
                        rev_mask,
                        *elem_bytes as usize,
                    );
                }
                Step::ArgReduceHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    reduced,
                    inner,
                    is_max,
                } => {
                    crate::vision_host::run_argreduce(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *reduced as usize,
                        *inner as usize,
                        *is_max,
                    );
                }
                Step::AxialRope2dHost {
                    src_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    hidden,
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    crate::vision_host::run_axial_rope2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *hidden as usize,
                        *end_x as usize,
                        *end_y as usize,
                        *head_dim as usize,
                        *num_heads as usize,
                        *theta,
                        *repeat_factor as usize,
                    );
                }
                Step::GruHost {
                    x,
                    w_ih,
                    w_hh,
                    b_ih,
                    b_hh,
                    h0,
                    dst,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    crate::vision_host::run_gru(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x as usize,
                        *w_ih as usize,
                        *w_hh as usize,
                        *b_ih as usize,
                        *b_hh as usize,
                        *h0 as usize,
                        *dst as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                    );
                }
                Step::RnnHost {
                    x,
                    w_ih,
                    w_hh,
                    bias,
                    h0,
                    dst,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    crate::vision_host::run_rnn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x as usize,
                        *w_ih as usize,
                        *w_hh as usize,
                        *bias as usize,
                        *h0 as usize,
                        *dst as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                        *relu,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_byte_off,
                    route_byte_off,
                    out_byte_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *sig_byte_off as usize,
                        *route_byte_off as usize,
                        *out_byte_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::UmapKnnHost {
                    pairwise_byte_off,
                    out_byte_off,
                    n,
                    k,
                } => {
                    crate::umap_knn_host::run_umap_knn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *pairwise_byte_off as usize,
                        *out_byte_off as usize,
                        *n as usize,
                        *k as usize,
                    );
                }
                Step::MsDeformAttnHost {
                    in_offs,
                    out_byte_off,
                    out_bytes,
                    attrs,
                } => {
                    crate::ms_deform_attn::run_ms_deform_attn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        in_offs,
                        *out_byte_off as usize,
                        *out_bytes as usize,
                        attrs,
                    );
                }
                Step::CollectiveHost {
                    name,
                    in_byte_off,
                    in_bytes,
                    out_byte_off,
                    out_bytes,
                    attrs,
                } => {
                    crate::collective_host::run_collective(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        name,
                        *in_byte_off as usize,
                        *in_bytes as usize,
                        *out_byte_off as usize,
                        *out_bytes as usize,
                        attrs,
                    );
                }
                Step::CustomHost {
                    name,
                    in_specs,
                    out_byte_off,
                    out_shape,
                    attrs,
                } => {
                    crate::custom_host::run_custom_host(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        name,
                        in_specs,
                        *out_byte_off as usize,
                        out_shape,
                        attrs,
                    );
                }
                Step::FftHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype_tag,
                } => {
                    crate::fft_host::run_fft1d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        *norm_tag,
                        fft_dtype_from_tag(*dtype_tag),
                    );
                }
                Step::FftQHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    scale_tag,
                } => {
                    crate::fft_host::run_fft1d_q(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        *norm_tag,
                        *scale_tag,
                    );
                }
                Step::ScanHost { desc } => {
                    crate::scan_host::run_scan(&self.arena, &dev.device, &dev.queue, desc);
                }
                Step::HostOp {
                    desc,
                    param_operands,
                } => {
                    crate::scan_host::run_host_op_with_cache(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        desc,
                        param_operands,
                        Some(&mut host_cache),
                    );
                }
                Step::CpuIndexing { thunk } => {
                    crate::scan_host::run_indexing(&self.arena, &dev.device, &dev.queue, thunk);
                }
                Step::ConcatHost {
                    dst_byte_off,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                    param_operands,
                } => {
                    crate::scan_host::run_concat_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off,
                        *outer,
                        *inner,
                        *total_axis,
                        inputs,
                        param_operands,
                        Some(&mut host_cache),
                    );
                }
                Step::ConcatHostPieces {
                    dst_byte_off,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                    starts,
                    param_operands,
                } => {
                    // Partial fill — other columns already on device; flush first.
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                        crate::backend::note_device_wait();
                        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    }
                    crate::scan_host::run_concat_host_pieces(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off,
                        *outer,
                        *inner,
                        *total_axis,
                        inputs,
                        starts,
                        param_operands,
                        /*clear=*/ false,
                    );
                }
                Step::TransposeHost {
                    in_byte_off,
                    out_byte_off,
                    in_dims,
                    out_dims,
                    in_strides,
                    param_operand,
                } => {
                    crate::scan_host::run_transpose_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        in_dims,
                        out_dims,
                        in_strides,
                        *param_operand,
                        Some(&mut host_cache),
                    );
                }
                Step::NarrowHost {
                    in_byte_off,
                    out_byte_off,
                    outer,
                    inner,
                    axis_in_size,
                    start,
                    axis_out_size,
                    param_operand,
                } => {
                    crate::scan_host::run_narrow_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        *outer,
                        *inner,
                        *axis_in_size,
                        *start,
                        *axis_out_size,
                        *param_operand,
                        Some(&mut host_cache),
                    );
                }
                Step::ExpandHost {
                    in_byte_off,
                    out_byte_off,
                    in_dims,
                    out_dims,
                    param_operand,
                } => {
                    crate::scan_host::run_expand_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        in_dims,
                        out_dims,
                        *param_operand,
                        Some(&mut host_cache),
                    );
                }
                Step::SpdHost {
                    op,
                    inputs,
                    out_shape,
                    out_byte_off,
                } => {
                    crate::spd_host::run_spd(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        op,
                        inputs,
                        out_shape,
                        *out_byte_off,
                    );
                }
                Step::WelchPeaksHost { .. }
                | Step::LogMelHost { .. }
                | Step::LogMelBackwardHost { .. } => {
                    unreachable!("tail-host audio ops run after GPU wait")
                }
                Step::Im2ColHost {
                    x_byte_off,
                    col_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                } => {
                    crate::im2col_host::run_im2col(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off as usize,
                        *col_byte_off as usize,
                        *n,
                        *c_in,
                        *h,
                        *w,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                    );
                }
                Step::Conv2dBackwardWeightHost {
                    x_byte_off,
                    dy_byte_off,
                    dw_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    groups,
                } => {
                    crate::conv_bwd_host::run_conv2d_backward_weight(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off,
                        *dy_byte_off,
                        *dw_byte_off,
                        *n,
                        *c_in,
                        *h,
                        *w,
                        *c_out,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                        *groups,
                    );
                }
                Step::Conv2dBackwardInputHost {
                    dy_byte_off,
                    w_byte_off,
                    dx_byte_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    groups,
                } => {
                    crate::conv_bwd_host::run_conv2d_backward_input(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dy_byte_off,
                        *w_byte_off,
                        *dx_byte_off,
                        *n,
                        *c_in,
                        *h,
                        *w_in,
                        *c_out,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                        *groups,
                    );
                }
                Step::RngNormalHost {
                    dst_byte_off,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    match opts.backend {
                        // On-device Philox / Zero — one-shot compute dispatch, no
                        // D2H→CPU→H2D bubble.
                        rlx_ir::RngBackend::Philox | rlx_ir::RngBackend::Zero => {
                            let zero = matches!(opts.backend, rlx_ir::RngBackend::Zero);
                            let (seed_lo, seed_hi) = if zero {
                                (0u32, 0u32)
                            } else {
                                let seed = rlx_ir::combine_seed(opts.seed, *key);
                                ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32)
                            };
                            let kern = if zero {
                                crate::kernels::rng_fill_zero_kernel(&dev.device)
                            } else {
                                crate::kernels::rng_normal_philox_kernel(&dev.device)
                            };
                            wgpu_rng_dispatch(
                                &dev.device,
                                &dev.queue,
                                &self.arena,
                                &mut host_cache,
                                kern,
                                *dst_byte_off,
                                *len,
                                *mean,
                                *scale,
                                seed_lo,
                                seed_hi,
                            );
                        }
                        // Ort / Bnns parity streams: host fill → H2D.
                        rlx_ir::RngBackend::Ort | rlx_ir::RngBackend::Bnns => {
                            crate::rng_host::run_rng_normal(
                                &self.arena,
                                &dev.device,
                                &dev.queue,
                                *dst_byte_off as usize,
                                *len as usize,
                                *mean,
                                *scale,
                                *key,
                                *op_seed,
                                opts,
                            );
                        }
                    }
                }
                Step::RngUniformHost {
                    dst_byte_off,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    match opts.backend {
                        rlx_ir::RngBackend::Philox | rlx_ir::RngBackend::Zero => {
                            let zero = matches!(opts.backend, rlx_ir::RngBackend::Zero);
                            let (seed_lo, seed_hi) = if zero {
                                (0u32, 0u32)
                            } else {
                                let seed = rlx_ir::combine_seed(opts.seed, *key);
                                ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32)
                            };
                            let kern = if zero {
                                crate::kernels::rng_fill_zero_kernel(&dev.device)
                            } else {
                                crate::kernels::rng_uniform_philox_kernel(&dev.device)
                            };
                            wgpu_rng_dispatch(
                                &dev.device,
                                &dev.queue,
                                &self.arena,
                                &mut host_cache,
                                kern,
                                *dst_byte_off,
                                *len,
                                *low,
                                *high,
                                seed_lo,
                                seed_hi,
                            );
                        }
                        rlx_ir::RngBackend::Ort | rlx_ir::RngBackend::Bnns => {
                            crate::rng_host::run_rng_uniform(
                                &self.arena,
                                &dev.device,
                                &dev.queue,
                                *dst_byte_off as usize,
                                *len as usize,
                                *low,
                                *high,
                                *key,
                                *op_seed,
                                opts,
                            );
                        }
                    }
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRender {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    dst_byte_off,
                    dst_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_render(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatPrepare {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    meta_len,
                    prep_byte_off,
                    prep_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_prepare(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRasterize {
                    prep_byte_off,
                    prep_len,
                    meta_byte_off,
                    meta_len,
                    dst_byte_off,
                    dst_len,
                    count,
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_rasterize(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *count as usize,
                        *width,
                        *height,
                        *tile_size,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRenderBackward {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    d_loss_byte_off,
                    d_loss_len,
                    packed_byte_off,
                    packed_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    crate::splat::run_gaussian_splat_render_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *d_loss_byte_off as usize,
                        *d_loss_len as usize,
                        *packed_byte_off as usize,
                        *packed_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                        *loss_grad_clip,
                        *sh_band,
                        *max_anisotropy,
                    );
                }
                _ => break,
            }
            // The step just rewrote the arena directly, so every cached
            // mirror entry is potentially stale (see `bypasses_host_cache`).
            if bypasses_host_cache {
                host_cache.clear();
            }
            step_i += 1;
            // Host paths stage with `queue.write_buffer` (no submit). Flush
            // deferred mirrors before the next GPU compute pass so kernels see
            // host results on discrete Vulkan/DX12. Reached by the host arms that
            // fall through to here, which is every cache-aware one (ConcatHost,
            // ExpandHost, NarrowHost, TransposeHost, HostOp, Conv2dHost, …). The
            // arm that `continue`s (the host `BufferCopy`) skips this door, so its
            // deferred write is reached by the next flush instead: the join's own
            // flush when the following step is a joined packed multiply, or the
            // host boundary in front of the next host reader of that slot.
            //
            // A joined copy no longer breaks the batch, so this door fires in
            // front of it too. What it owes there is the device poll that a host
            // step owes the device work after it; it cannot supply a flush, because
            // a joined copy is only reachable with no deferred write pending — a
            // dirty mirror forces the copy's own route to `Host`, which makes it a
            // boundary here and leaves the door unfired. A joined *packed multiply*
            // is the arm that still needs the flush: nothing in its route consults
            // the mirror.
            if step_i < self.schedule.len() {
                // A joined packed multiply or a joined device-route copy is the
                // next batch's device work: it reads what this host step wrote, so
                // the mirror reaches the device (and the device is polled) before
                // the batch runs — the same door a plain device step goes through.
                if !self.step_breaks_batch(step_i, &scale, one_op_per_pass, &host_cache, dev) {
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                    }
                    crate::backend::note_device_wait();
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    // Keep mirrors until GPU actually runs (cleared when
                    // `pass_dispatched` on the next host sync).
                }
            } else if host_cache.has_deferred_writes() {
                let mut a = crate::host_stage::WgpuArena {
                    arena: &self.arena,
                    device: &dev.device,
                    queue: &dev.queue,
                    size_bytes: 0,
                };
                host_cache.flush_to_device(&mut a);
                crate::backend::note_device_wait();
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            } else {
                crate::backend::note_device_wait();
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
        }

        // The loop's own exit: the schedule's end, read while the schedule is
        // still in hand, so the interval still open closes over the steps it
        // covered rather than over an empty cursor range.
        report.end_of_schedule(step_i, &self.schedule);

        if host_cache.has_deferred_writes() {
            let mut a = crate::host_stage::WgpuArena {
                arena: &self.arena,
                device: &dev.device,
                queue: &dev.queue,
                size_bytes: 0,
            };
            host_cache.flush_to_device(&mut a);
            crate::backend::note_device_wait();
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        }

        self.debug_scans_after_run(dev);

        if rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") {
            if !self.static_once_done && !self.static_once_steps.is_empty() {
                self.static_once_done = true;
            }
            return self
                .graph
                .outputs
                .iter()
                .map(|&id| {
                    let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    vec![0.0; n]
                })
                .collect();
        }
        if !self.static_once_done && !self.static_once_steps.is_empty() {
            self.static_once_done = true;
        }
        report.open(Phase::Readback, step_i, &self.schedule);
        let out_ids: Vec<_> = self.graph.outputs.clone();
        read_f32_many_pooled(
            &self.arena,
            &dev.device,
            &dev.queue,
            &out_ids,
            &mut self.readback_staging,
        )
    }
}

/// One-shot on-device Philox / Zero RNG fill (no host round-trip): binds a
/// 256-byte-aligned arena window plus a params uniform, dispatches `kern`
/// (`rng_philox.wgsl`), then marks the region GPU-dirty so later host reads
/// re-fetch. Mirrors the `arena_copy` self-contained dispatch.
#[allow(clippy::too_many_arguments)]
fn wgpu_rng_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    arena: &Arena,
    host_cache: &mut rlx_gpu_host::HostTensorCache,
    kern: &crate::kernels::Kernel,
    dst_byte_off: u32,
    len: u32,
    a: f32,
    b: f32,
    seed_lo: u32,
    seed_hi: u32,
) {
    if len == 0 {
        return;
    }
    let dst = dst_byte_off as u64;
    let max_binding = device.limits().max_storage_buffer_binding_size;
    let hi = dst + (len as u64) * 4;
    let mut base = (dst / 256) * 256;
    let span = hi.saturating_sub(base).max(1);
    let mut size = span.div_ceil(256) * 256;
    size = size.max(256).min(max_binding);
    if base.saturating_add(size) > arena.size as u64 {
        base = (arena.size as u64).saturating_sub(size);
        base = (base / 256) * 256;
    }
    let p = crate::kernels::RngParams {
        n: len,
        out_off: ((dst - base) / 4) as u32,
        a,
        b,
        seed_lo,
        seed_hi,
        _p0: 0,
        _p1: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu rng uniform"),
        size: std::mem::size_of::<crate::kernels::RngParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = bind_arena_window(device, kern, arena, base, size, &u);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu rng"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu rng pass"),
            ..Default::default()
        });
        pass.set_pipeline(&kern.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(len, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    crate::backend::note_device_submit();
    queue.submit(std::iter::once(enc.finish()));
    crate::backend::note_device_wait();
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    host_cache.invalidate(dst as usize);
}
