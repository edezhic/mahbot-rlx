// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a single-row packed arm multiplies its activation with, per scheme and per
//! build — the one place that fact is written down.
//!
//! It is read at three resolutions, each narrowing the one before: [`activation`] is the
//! scheme's answer on this build, [`fused_m1_quantizes_activation`] is that answer as a
//! yes/no — the question the cache-state predicate asks, since diverting a decode to an
//! arm that rounds would change its answer — and `rounds_to_q8k_at` narrows it to the call
//! and to the Q8_K form the harnesses have an oracle for. None of them keeps a list of its
//! own.

#[cfg(any(test, feature = "backend-tests"))]
use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;

/// Whether the x86 `int8-gemv` kernel can run on this host: its module's cfg is
/// compile-time, its AVX2/FMA/F16C probe is not, and the kernel stands itself down
/// when the probe fails — so the routing has to ask both.
fn int8_gemv_available() -> bool {
    #[cfg(all(feature = "int8-gemv", rlx_cpu_x86))]
    {
        crate::int8_gemv::available()
    }
    #[cfg(not(all(feature = "int8-gemv", rlx_cpu_x86)))]
    {
        false
    }
}

/// What a single-row packed arm multiplies its activation with on this build.
pub(crate) enum Activation {
    /// The f32 row as given.
    F32,
    /// One Q8_K block per 256 elements — llama.cpp's K-quant form.
    Q8K,
    /// 32-element int8 blocks with their own scales, `int8-gemv`'s form for the
    /// 32-element schemes.
    Int8Blocks,
}

/// The activation `scheme`'s single-row arm uses, read from the same cfgs, probes
/// and switches that gate the kernels themselves.
///
/// Every integer arm that rounds to Q8_K: Q4_K's own default arm, the six-bit and
/// eight-bit routes where they are compiled, and `int8-gemv`'s Q4_K/Q5_K/Q6_K arms (the
/// kernel there is `int8_gemv::q8k_row`). That kernel's 32-element arms — Q8_0, Q4_0,
/// Q4_1 — round to int8 blocks instead; Q2_K, Q3_K and Q8_K it does not cover.
/// Every other arm runs `gguf_matmul_bt_m1_rows_generic` over f32.
///
/// `RLX_Q4K_EXACT_GEMV`, `RLX_Q6K_EXACT_GEMV` and `RLX_Q8_0_EXACT_GEMV` ask for the
/// f32-activation kernel on purpose, so each of those schemes stops rounding while its own
/// switch is set. The six-bit and eight-bit switches reach their arms only where those
/// routes are compiled (aarch64): on x86-64 `int8-gemv` takes both widths first and does
/// not consult either.
///
/// The `int8-gemv` arms ask [`int8_gemv_available`], not just its cfg: a host that
/// compiled the kernel but cannot run it takes the f32 route, and a resolution that
/// named a rounding there would make an oracle expect one the arm never performs.
/// The probe is cached, so this cannot move within a process.
///
/// It is the scheme's answer, not the call's: the parallel arms have a gate of their own
/// ([`crate::gguf_matmul::m1_parallel_takes`]), so a host or a shape they refuse multiplies
/// those widths in f32 while this still names Q8_K. For routing that is the safe direction;
/// a harness asks [`rounds_to_q8k_at`] instead.
pub(crate) fn activation(scheme: QuantScheme) -> Activation {
    match scheme {
        QuantScheme::GgufQ4K if crate::gguf_matmul::q4k_exact_gemv() => Activation::F32,
        QuantScheme::GgufQ4K => Activation::Q8K,
        QuantScheme::GgufQ6K if crate::gguf_m1::COMPILED => {
            if crate::gguf_q6k::exact_asked() {
                Activation::F32
            } else {
                Activation::Q8K
            }
        }
        QuantScheme::GgufQ8_0 if crate::gguf_m1::COMPILED => {
            if crate::gguf_q8_0::exact_asked() {
                Activation::F32
            } else {
                Activation::Q8K
            }
        }
        QuantScheme::GgufQ5K | QuantScheme::GgufQ6K if int8_gemv_available() => Activation::Q8K,
        QuantScheme::GgufQ8_0 | QuantScheme::GgufQ4_0 | QuantScheme::GgufQ4_1
            if int8_gemv_available() =>
        {
            Activation::Int8Blocks
        }
        _ => Activation::F32,
    }
}

/// Whether the single-row arm rounds the activation at all for `scheme` on this
/// build — the scheme-level question the cache-state predicate asks, before any call's
/// gates.
pub(crate) fn fused_m1_quantizes_activation(scheme: QuantScheme) -> bool {
    !matches!(activation(scheme), Activation::F32)
}

/// Whether a single-row call of `scheme` at this shape multiplies a Q8_K activation on
/// this host: [`activation`] narrowed to the arm the call reaches, so a harness's oracle
/// is the route that ran rather than the one the scheme names. The eight-bit call's answer
/// is [`crate::gguf_q8_0::takes`], narrower than the shared arm — it also gates the exact
/// handle, reproducible mode and whole rows of weight.
///
/// The answer describes the shipped dispatch: cache off or ignored, reproducible mode off.
/// Both ambient states move a call off the route this names — a thrashing cache sends a
/// rounding arm that keeps it (Q4_K, Q6_K) to the f32 slab, and reproducible mode sends
/// every one of them there — while the eight-bit arm's taken calls never reach the cache
/// at all. A harness reading this pins both off, as the ones here do.
#[cfg(any(test, feature = "backend-tests"))]
pub(crate) fn rounds_to_q8k_at(k: usize, n: usize, w_len: usize, scheme: QuantScheme) -> bool {
    // Every arm that rounds to Q8_K quantises the activation a whole super-block at a
    // time, so a `k` that is not a multiple of QK_K falls to an f32 route.
    if !matches!(activation(scheme), Activation::Q8K) || !k.is_multiple_of(QK_K) {
        return false;
    }
    match scheme {
        // Q4_K's sequential single-row kernel quantises too (`q4k_gemv_rowmajor`), so
        // the pool gate does not reach it.
        QuantScheme::GgufQ4K => true,
        QuantScheme::GgufQ8_0 => crate::gguf_q8_0::takes(k, n, w_len),
        // The six-bit route and `int8-gemv`'s arms live in the parallel single-row arm,
        // so its own gate decides whether the call reaches them.
        _ => crate::gguf_matmul::m1_parallel_takes(crate::gguf_matmul::packed_blocks(
            k, n, w_len, scheme,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ends of the resolution that are surprising: Q4_K rounds on every host
    /// (its arm quantises unless the exact switch is set), Q8_K on none (no integer
    /// arm anywhere), the six-bit arm wherever its integer dot is compiled in, and
    /// the `int8-gemv` arms — whose K-quants round to the same Q8_K form as the
    /// rest, which is what lets a harness hold them to its Q8_K oracle.
    ///
    /// Neither resolution can be pinned whole without a second copy of the routing
    /// it mirrors; the exact switches and reproducible mode are what this masks off,
    /// on this thread only ([`rlx_ir::env::scoped`]) so a shell's value or a sibling
    /// test setting one for its own window cannot be read here.
    #[test]
    fn the_rounding_predicate_names_the_integer_arms() {
        let _q4k = rlx_ir::env::scoped("RLX_Q4K_EXACT_GEMV", "0");
        let _q6k = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "0");
        let _q8_0 = rlx_ir::env::scoped("RLX_Q8_0_EXACT_GEMV", "0");
        let _reproducible = rlx_ir::env::scoped("RLX_REPRODUCIBLE", "0");
        // A weight's packed length from its block count, taken from the scheme so a
        // block-layout change cannot leave these shapes describing an older one.
        let q8_0_len = |blocks: usize| blocks * QuantScheme::GgufQ8_0.gguf_block_bytes() as usize;
        assert!(fused_m1_quantizes_activation(QuantScheme::GgufQ4K));
        assert!(matches!(activation(QuantScheme::GgufQ4K), Activation::Q8K));
        assert!(!fused_m1_quantizes_activation(QuantScheme::GgufQ8K));
        if crate::gguf_m1::COMPILED {
            assert!(matches!(activation(QuantScheme::GgufQ6K), Activation::Q8K));
            assert!(matches!(activation(QuantScheme::GgufQ8_0), Activation::Q8K));
        }
        // A shape the arm refuses is not described as rounding: the edit that would make
        // the eight-bit arm unconditional. The taken case is pinned end to end by
        // `tests/gguf_single_row_cache_pressure.rs`, whose oracle is this predicate.
        assert!(!rounds_to_q8k_at(
            256,
            1,
            q8_0_len(8),
            QuantScheme::GgufQ8_0
        ));
        // A `k` that is not a whole super-block reaches no Q8_K arm, whatever the scheme:
        // the six-bit route's own super-block gate is the one this has to carry, since the
        // arm below it is the shared parallel one. The weight is a six-bit one of the
        // whole super-blocks `k` covers.
        assert!(!rounds_to_q8k_at(
            1024 + 32,
            64,
            64 * 4 * QuantScheme::GgufQ6K.gguf_block_bytes() as usize,
            QuantScheme::GgufQ6K
        ));
        // Reproducible mode asks for one route whatever the host: the eight-bit arm
        // stands down, so the route cannot depend on core count or `dotprod`.
        {
            let _reproducible = rlx_ir::env::scoped("RLX_REPRODUCIBLE", "1");
            assert!(!rounds_to_q8k_at(
                1024,
                64,
                q8_0_len(64 * 32),
                QuantScheme::GgufQ8_0
            ));
        }
        if int8_gemv_available() {
            for scheme in [
                QuantScheme::GgufQ4K,
                QuantScheme::GgufQ5K,
                QuantScheme::GgufQ6K,
            ] {
                assert!(matches!(activation(scheme), Activation::Q8K), "{scheme:?}");
            }
            for scheme in [
                QuantScheme::GgufQ8_0,
                QuantScheme::GgufQ4_0,
                QuantScheme::GgufQ4_1,
            ] {
                assert!(
                    matches!(activation(scheme), Activation::Int8Blocks),
                    "{scheme:?}"
                );
            }
        }
    }
}
