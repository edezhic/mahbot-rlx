// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a single-row packed arm multiplies its activation with, per scheme and per
//! build — the one place that fact is written down.
//!
//! Two callers need it at different resolutions: the cache-state predicate asks
//! whether diverting a decode to the fused arm would change the answer, and the
//! single-row harnesses ask which oracle to hold an arm to. Both read [`activation`]
//! rather than keeping a list of their own.

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
/// Every K-quant integer arm rounds to Q8_K: Q4_K's own default arm, the six-bit
/// route where it is compiled, and `int8-gemv`'s Q4_K/Q5_K/Q6_K arms (the kernel
/// there is `int8_gemv::q8k_row`). That kernel's 32-element arms — Q8_0, Q4_0,
/// Q4_1 — round to int8 blocks instead; Q2_K, Q3_K and Q8_K it does not cover.
/// Every other arm runs `gguf_matmul_bt_m1_rows_generic` over f32.
///
/// `RLX_Q4K_EXACT_GEMV` and `RLX_Q6K_EXACT_GEMV` ask for the f32-activation kernel
/// on purpose, so those two schemes do not round while either is set. The six-bit
/// switch reaches the arm only where that route is compiled (aarch64): on x86-64
/// `int8-gemv` takes the width first and does not consult it.
///
/// The `int8-gemv` arms ask [`int8_gemv_available`], not just its cfg: a host that
/// compiled the kernel but cannot run it takes the f32 route, and a resolution that
/// named a rounding there would make an oracle expect one the arm never performs.
/// The probe is cached, so this cannot move within a process.
///
/// It is the scheme's answer, not the call's: the six-bit route also needs a pool of
/// more than one worker and at least 32 blocks of weight, so a one-worker host
/// multiplies that width in f32 and this still names Q8_K. For routing that is the
/// safe direction; a harness building an oracle from it is right only where those
/// hold.
pub(crate) fn activation(scheme: QuantScheme) -> Activation {
    match scheme {
        QuantScheme::GgufQ4K if crate::gguf_matmul::q4k_exact_gemv() => Activation::F32,
        QuantScheme::GgufQ4K => Activation::Q8K,
        QuantScheme::GgufQ6K if crate::gguf_q6k::COMPILED => {
            if crate::gguf_q6k::exact_asked() {
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
/// build, i.e. whether diverting a decode GEMV to it for speed would change the
/// answer.
pub(crate) fn fused_m1_quantizes_activation(scheme: QuantScheme) -> bool {
    !matches!(activation(scheme), Activation::F32)
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
    /// it mirrors; the exact switches are what this masks off, on this thread only
    /// ([`rlx_ir::env::scoped`]) so a sibling test setting either to `1` for its own
    /// window cannot be read here.
    #[test]
    fn the_rounding_predicate_names_the_integer_arms() {
        let _q4k = rlx_ir::env::scoped("RLX_Q4K_EXACT_GEMV", "0");
        let _q6k = rlx_ir::env::scoped("RLX_Q6K_EXACT_GEMV", "0");
        assert!(fused_m1_quantizes_activation(QuantScheme::GgufQ4K));
        assert!(matches!(activation(QuantScheme::GgufQ4K), Activation::Q8K));
        assert!(!fused_m1_quantizes_activation(QuantScheme::GgufQ8K));
        if crate::gguf_q6k::COMPILED {
            assert!(matches!(activation(QuantScheme::GgufQ6K), Activation::Q8K));
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
