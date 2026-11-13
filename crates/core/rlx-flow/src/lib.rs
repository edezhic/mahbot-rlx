// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block assembly-line API for RLX model builders.
//!
//! # Extending rlx from a downstream crate
//!
//! A novel architecture block can live **entirely in a downstream crate**
//! (e.g. a model crate in `rlx-models`) — no edit to this crate's closed
//! [`FlowStage`] enum, and no republish gate. Implement [`LayerStage`] for your
//! block, composing primitives through the [`FlowCtx`] emission surface, then
//! drop it into any flow:
//!
//! ```ignore
//! use rlx_flow::prelude::*;
//!
//! struct MyBlock { /* config, weight keys */ }
//! impl LayerStage for MyBlock {
//!     fn name(&self) -> &str { "my_block" }
//!     fn emit_layer(&self, ctx: &mut FlowCtx<'_>, input: FlowValue)
//!         -> anyhow::Result<(FlowValue, StageArtifacts)>
//!     {
//!         // compose primitives via ctx (load_param + HirMut builders) …
//!         # unimplemented!()
//!     }
//! }
//!
//! let flow = ModelFlow::new("my_model")
//!     .input("x", shape)
//!     .layer_stage(MyBlock { /* … */ });   // <- extension front door
//! ```
//!
//! Because the block lowers to ordinary primitives (not an opaque `Op::Custom`),
//! it stays visible to fusion and hits the fast path on every backend. For a new
//! *operation* rather than a *block*, see the two op seams in `rlx-ir`:
//! `Graph::custom_op` + [`rlx_ir::OpExtension`] (register a kernel, or an
//! [`OpExtension::lower`](rlx_ir::OpExtension::lower) decompose-to-primitives
//! rule so it runs on every backend with no kernel), and `Graph::custom_fn` for
//! a subgraph op with custom autodiff.

pub mod blocks;
mod composite;
mod context;
mod dsl;
pub mod escape;
mod execution;
mod extension;
mod flow;
#[cfg(feature = "formats")]
pub mod formats;
mod layer;
pub mod packed_attn;
mod plugin;
mod profile;
mod recipe;
pub mod rope;
mod side;
mod stage;
mod stage_contract;
mod stage_interfaces;
pub mod storage;
pub mod stream;
mod value;
mod weight;

pub mod prelude;

pub use blocks::RopeTablesStage;
pub use blocks::{
    BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle, ClsTokenPoolStage, FfnActivation,
    GeluFfnStage, NomicEncoderLayerSpec, NomicEncoderLayerStage, Qwen3DecodeLayerSpec,
    Qwen3DecoderSpec, Qwen3DecoderStage, VitSelfAttnSpec, dinov2_layer_fused, dit_ada_gated_linear,
    nomic_vision_layer_fused, transformer_encoder_layer,
};
pub use composite::LayerComposition;
// `FlowCtx` is the emission surface a downstream `LayerStage` receives, so it
// must be nameable outside the crate for the extension seam to be usable.
pub use context::{DecodeBindings, FlowCtx, FlowState, GdnInputSlots, f64_format};
pub use escape::Emit;
pub use execution::{ExecutionPreset, ModelExecutionConfig};
pub use extension::FlowExtensionPlan;
pub use flow::{BuiltModel, ModelFlow};
pub use layer::LayerStack;
pub use plugin::{PluginStage, plugin, plugin_named};
pub use profile::{
    BackendOverrides, CompileProfile, CpuBackendProfile, FusionPolicyKind, FusionProfile,
    FusionTargetKind, MetalBackendProfile, MixedPrecisionKind, PassProfile, PrecisionKind,
    PrecisionProfile, ProfileMode,
};
pub use recipe::ModelRecipe;
pub use rope::{
    Llama3Scaling, YarnScaling, build_default_tables, build_mrope_tables, build_mrope_text_tables,
    build_tables, build_yarn_tables, default_inv_freq, inv_freq_with_factors,
    llama3_scaled_inv_freq, mrope_row_for_sections, mrope_row_for_sections_ex,
    mrope_section_for_pair, mrope_sections4, ntk_scaled_inv_freq, yarn_mscale,
    yarn_scaled_inv_freq,
};
pub use side::SideOutputs;
pub use stage::FlowStage;
pub use stage_contract::{BlockAsLayer, DynStage, LayerStage, StageArtifacts};
pub use stage_interfaces::{AttentionStage, FfnStage, KvCacheContract, NormStage};
pub use storage::{WEIGHT_STORAGE_ENV, WeightStorage};
pub use stream::{
    DualStreamStage, LoadStreamStage, StoreStreamStage, dual_stream_stage, id as stream_id,
};
pub use value::FlowValue;
pub use weight::{
    Aliased, CloningSource, Fallback, MapWeights, Mapped, Prefixed, Renamed, TensorLookup, Tracked,
    WeightReport, WeightSource, WeightSourceExt, missing_weight, transpose_2d,
};

use std::collections::HashMap;

/// Compatibility shim: packed GGUF matmul weights (used by some model loaders).
#[derive(Debug, Clone, Default)]
pub struct GgufPackedParams {
    pub linears: HashMap<String, GgufPackedLinear>,
}

impl GgufPackedParams {
    pub fn get_linear(&self, key: &str) -> Option<&GgufPackedLinear> {
        self.linears.get(key)
    }
}

/// One packed linear weight: quantized bytes + bias.
#[derive(Debug, Clone)]
pub struct GgufPackedLinear {
    pub w_q: PackedBlob,
    pub scheme: rlx_ir::quant::QuantScheme,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: Vec<f32>,
}

/// A packed (quantised) weight blob: the bytes, or — for a source that binds them at
/// session time instead — the length they will have.
///
/// The length is what the parameter is declared at: `FlowCtx::linear` makes its shape `[len]`
/// from this blob and nothing else, so a source that carries no bytes has no other channel to
/// declare the parameter's size. It has to be the file's own tensor length. A source must not
/// derive it from the scheme's block table: that table is rounded (`QuantScheme`'s
/// bits-per-element figures are tenths, and Q6_K is 210 bytes per 256 elements where the table
/// says 66), and a wrong length declares the parameter at the wrong size.
#[derive(Debug, Clone)]
pub struct PackedBlob {
    /// The bytes, when the source carried them: `None` for a declaration, which is what
    /// makes the two cases distinguishable at every call site that has to join or upload
    /// them rather than declare them.
    bytes: Option<Vec<u8>>,
    /// The blob's byte length: the carried bytes' own length, or the declared one. It is a
    /// field rather than a getter because it is the whole reason the type exists.
    pub len: usize,
}

impl PackedBlob {
    /// A blob of `len` bytes that this source does not carry: the parameter is declared
    /// at `len` and its bytes are bound at session time, from wherever the source knows
    /// them to be. Nothing is allocated, so a build that declares a checkpoint's worth of
    /// weights pays nothing for them.
    #[must_use]
    pub fn declared(len: usize) -> Self {
        Self { bytes: None, len }
    }

    /// The bytes, when this blob carries them — `None` for a declaration, so a caller that
    /// has to concatenate or upload them cannot forget that a source may not have handed
    /// them over. The declared length stands either way.
    #[must_use]
    pub fn carried_bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }

    /// The bytes as the build hands them to the session — empty for a declaration, whose
    /// parameter is bound from wherever the source said they are.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes.unwrap_or_default()
    }
}

impl From<Vec<u8>> for PackedBlob {
    fn from(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len(),
            bytes: Some(bytes),
        }
    }
}

/// A packed **MoE expert bank**: `num_experts` contiguous `[out_dim, in_dim]`
/// quant slabs in one blob.
///
/// The grouped counterpart of [`GgufPackedLinear`], which cannot describe this
/// because it has no expert count — a 3-D `[E, out, in]` tensor read through
/// `take_packed` would report `out_dim = E`. Served by
/// [`WeightSource::take_packed_bank`] and consumed by
/// `Graph::dequant_grouped_matmul_packed`.
///
/// The layout is GGUF's native order for `ffn_*_exps.weight`, so a loader hands
/// the bytes over untouched.
#[derive(Debug, Clone)]
pub struct GgufPackedBank {
    pub w_q: Vec<u8>,
    pub scheme: rlx_ir::quant::QuantScheme,
    pub num_experts: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}
