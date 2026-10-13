// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed wgpu runtime options loaded once from the `RLX_*` registry.

use std::sync::{OnceLock, RwLock};

/// wgpu backend options (conv thresholds, dumps, coop-matrix).
#[derive(Debug, Clone)]
pub struct WgpuRuntimeConfig {
    pub gdn_host: bool,
    pub im2col_min_spatial: usize,
    pub im2col_min_k: usize,
    pub im2col_min_cout: usize,
    pub tiled_min_spatial: usize,
    pub dump_nodes: bool,
    pub dump_nodes_limit: usize,
    pub dump_tail: bool,
    pub dump_inputs: bool,
    pub schedule: bool,
    pub large_buffers: bool,
    pub print_limits: bool,
    pub matmul_f32_only: bool,
    pub f16_weights: bool,
    pub no_tiled_conv: bool,
    /// The conv-routes switch (`RLX_WGPU_CONV_IM2COL`, default on) *as read when
    /// this snapshot was taken*. The router does not read it — `conv_routes_enabled()`
    /// reads the env live, so the switch is not frozen at snapshot time, exactly as
    /// `no_tiled_conv` above relates to the tiled route. Kept rather than deleted as
    /// reader-less: it is part of the public struct upstream also ships.
    pub conv_im2col: bool,
    pub debug: bool,
}

impl Default for WgpuRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl WgpuRuntimeConfig {
    pub fn from_env() -> Self {
        use rlx_ir::env_registry as reg;
        // Conv-route floors. The spatial floor's default is 100, down from 800: it admits
        // the whole band [100, 800), and the three convolutions of our own graphs that sit
        // in it all win there — the decoder entry conv (output 100, the floor itself; the
        // comparison is inclusive) at 282.4 → 258.7 ms of its stage's step (an earlier
        // five-round set: 281.0 → 260.0), its device pass 28.76 → 6.45 ms and total device
        // time 223.6 → 200.6 ms; the encoder's third input conv (output 208, kernel 3,
        // stride 2) at 82.0 → 71.1 ms; and the long decoder input (output 744, ≈14.9 s of
        // speech) at 1.72–1.75 → 1.62–1.63 s, spread 0.01–0.03 s. Outputs of the measured
        // cells stay bit-identical (these shapes share one accumulation order, which is not
        // a general guarantee), the memory guard passes, submissions/waits stay 18/24 and
        // the other seven cells' schedules are byte-identical. Those readings are taken *at*
        // 100: 800 was the only value sampled below the old 2048, with nothing sampled
        // between, so they are the evidence rather than an envelope the new default sits in.
        //
        // The band opens wholesale, and its two halves are not the same swap. Below 256 it
        // is direct convolution → accelerated — the tiled route's own floor is 256, so it
        // never served those shapes. From 256 up to 800, for a shape the relabel did serve,
        // it is the tiled/mirrored relabel → accelerated: the accelerated output-size floor
        // now sits below the tiled route's own, so the relabel leaves the sub-band it was
        // written for. It does not disappear: above its own floor it still serves shapes
        // that fail a *non-spatial* accelerated gate — channels under the channel floor (the
        // body's one-output-channel projection) or the term count under the term-count
        // floor, a weight parked in the separate weight buffer, a panel past the reserved
        // scratch or the column cap, an arena that does not bind whole — and nothing moves
        // at or above 800. The sub-band the relabel vacates ([256, 800)) belongs to other
        // graphs: the shapes that lose it are unit-height lifts clearing every remaining
        // accelerated gate, and of our three measured shapes the two under the relabel's own
        // 256 floor (the entry conv at 100 and the encoder's third input conv at 208) were
        // never served by it. Shapes of other graphs sitting in the band move unmeasured —
        // the accepted cost of going by the floor rather than by a rule derived from what the
        // two routes do at a given shape.
        //
        // Two consequences the move carries rather than implies. (1) Execution under an
        // active extent: the accelerated step is not extent-safe and the hint is
        // all-or-nothing per graph, so a graph whose convolution moves to it loses that
        // mode entirely, not just for the moved step. It is not a value difference — the
        // run proceeds at the full compiled extent, with no compile-time refusal, no route
        // change, no host fallback and no wrong result — and by default it is silent: the
        // only signal is one stderr line behind a debug-classified switch whose registry
        // summary describes something else, and the executable's "can honour the hint"
        // capability is a static constant that cannot express the per-graph veto. Nothing on
        // our own shipped path runs in that mode (it is a run-time hint set on an
        // already-compiled executable; only the bucketed-cache helpers set it in-tree, and
        // they have no in-tree caller), so the loss falls on users of that mode. (2) The
        // memory plan: the reservation for the accelerated route's patch panel is derived
        // from a convolution's shape, so in graphs other than ours the plan changes wherever
        // a newly admitted shape passes the reservation's own base. It is not untouched —
        // but in the measured cells the peak does not grow (RSS 864–1009 → 863–921 MiB,
        // wired 3.5–3.8 → 3.6–3.7 GiB).
        //
        // The device-free route-decision checks derive their fixtures from the effective
        // floors, so none of them witnesses this value: the assertions that do witness it
        // sit behind `backend-tests` and behind the `#[ignore]`d measurement harness, whose
        // own doc states how a fixture reaches the tiled route at the shipped default. The
        // 1×1 route's own bar is `crate::backend::POINTWISE_MIN_COL_ELEMS`, which this
        // floor does not move.
        Self {
            gdn_host: reg::flag("RLX_WGPU_GDN_HOST"),
            im2col_min_spatial: reg::parse_or("RLX_WGPU_IM2COL_MIN_SPATIAL", 100usize),
            im2col_min_k: reg::parse_or("RLX_WGPU_IM2COL_MIN_K", 256usize),
            im2col_min_cout: reg::parse_or("RLX_WGPU_IM2COL_MIN_COUT", 64usize),
            tiled_min_spatial: reg::parse_or("RLX_WGPU_TILED_MIN_SPATIAL", 256usize),
            dump_nodes: reg::flag("RLX_WGPU_DUMP_NODES"),
            dump_nodes_limit: reg::parse_or("RLX_WGPU_DUMP_NODES_LIMIT", 40usize),
            dump_tail: reg::flag("RLX_WGPU_DUMP_TAIL"),
            dump_inputs: reg::flag("RLX_WGPU_DUMP_INPUTS"),
            schedule: reg::flag("RLX_WGPU_SCHEDULE"),
            large_buffers: reg::flag("RLX_WGPU_LARGE_BUFFERS"),
            print_limits: reg::flag("RLX_WGPU_PRINT_LIMITS"),
            matmul_f32_only: reg::flag("RLX_WGPU_MATMUL_F32_ONLY"),
            f16_weights: reg::flag("RLX_WGPU_F16_WEIGHTS"),
            no_tiled_conv: reg::flag("RLX_WGPU_NO_TILED_CONV"),
            conv_im2col: reg::flag_or("RLX_WGPU_CONV_IM2COL", true),
            debug: reg::flag("RLX_WGPU_DEBUG"),
        }
    }
}

static CONFIG: OnceLock<RwLock<WgpuRuntimeConfig>> = OnceLock::new();

fn map() -> &'static RwLock<WgpuRuntimeConfig> {
    CONFIG.get_or_init(|| RwLock::new(WgpuRuntimeConfig::from_env()))
}

pub fn runtime_config() -> WgpuRuntimeConfig {
    map().read().expect("wgpu config").clone()
}

pub fn reload_runtime_config() {
    *map().write().expect("wgpu config") = WgpuRuntimeConfig::from_env();
}

pub fn install_runtime_config(cfg: WgpuRuntimeConfig) {
    *map().write().expect("wgpu config") = cfg;
}
