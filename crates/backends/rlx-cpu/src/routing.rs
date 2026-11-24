// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The routing facts the harnesses need — what a packed arm actually computes, so
//! their oracle is the crate's own resolution, and which multi-row route ran — derived
//! from the crate so there is no second list to drift.
//!
//! Compiled only under `backend-tests`, which is off in every published build: a
//! `pub(crate)` item cannot be re-exported as `pub`, hence the wrappers.
use rlx_ir::quant::QuantScheme;

/// Whether a single-row integer route is compiled in — aarch64 only, so the exact
/// switches are no-ops elsewhere and a harness measuring one must say which host it
/// measured on. The same fact the routes themselves read
/// ([`crate::gguf_m1::COMPILED`]).
pub const SINGLE_ROW_ROUTE_COMPILED: bool = crate::gguf_m1::COMPILED;

/// Whether an eight-bit single-row call of this shape is taken whatever the dequant cache
/// holds — the fact an oracle's activation rests on.
pub fn q8_0_arm_ignores_cache(k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> bool {
    crate::gguf_matmul::q8_0_arm_ignores_cache(k, n, w_len, scheme)
}

/// Whether the single-row arm rounds the activation at all.
pub fn quantizes_activation(scheme: QuantScheme) -> bool {
    crate::gguf_activation::fused_m1_quantizes_activation(scheme)
}

/// Whether it rounds to Q8_K — the only rounding this crate has an oracle for.
pub fn rounds_to_q8k(scheme: QuantScheme) -> bool {
    matches!(
        crate::gguf_activation::activation(scheme),
        crate::gguf_activation::Activation::Q8K
    )
}

/// [`rounds_to_q8k`] for the call's shape, so a harness's oracle is the arm the call
/// reaches rather than the one the scheme names.
pub fn rounds_to_q8k_at(k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> bool {
    crate::gguf_activation::rounds_to_q8k_at(k, n, w_len, scheme)
}

/// Why the multi-row tiled route does not take a shape on this host, or `None` when it
/// does: a harness labels the arm it timed with this, so the ratio names the route that
/// ran.
pub fn tiled_refusal(m: usize, k: usize, n: usize, scheme: QuantScheme) -> Option<&'static str> {
    crate::gguf_tiled::refusal(m, k, n, scheme)
}
