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
        // Conv-route floors. The spatial floor's default is 1, down from 100: the
        // materialising im2col+GEMM route used to require an output spatial extent
        // (`h_out * w_out`, the output length for a unit-height lift) of at least 100, and
        // now admits every extent a conv can have, 1 being the shortest. The spatial bar
        // is one of four — the K (256), channel (64) and groups floors still gate the
        // im2col route, and the relabel tests neither the K nor the channel floor (it
        // requires `groups == 1` as well) — so what this moves is the extent alone, never
        // a shape the other three refuse.
        //
        // Five shape families are what that newly admits in the product. The synthesis
        // front's own graph — priced by `tests/synthesis_front_route.rs`, which carries
        // the protocol, in one binary with both arms through this key — carries the
        // first three: the wave transformer's `pre_conv` at every window of 72 frames or
        // fewer, `dac.entry` at a part of 24 frames or fewer, and the stage-3 residual
        // lifts at a part of three frames or fewer:
        //
        //   frames    1      2      3      8     16     18     20
        //   @100  18.80  32.26  40.00  28.63  48.07  52.57  57.33
        //   @1    11.59  13.25  15.94  20.76  34.03  38.11  41.08
        //   delta  7.27  18.99  24.08   7.85  14.03  14.48  16.35
        //
        //   frames   22     23     24     26     32     42     60     72
        //   @100  38.42  39.62  41.04  23.54  27.65  33.12  44.64  53.02
        //   @1    20.90  24.09  21.03  23.00  26.71  32.13  43.47  50.80
        //   delta 17.55  15.53  19.93   0.67   0.99   1.02   1.10   2.29
        //
        // `delta` is the median of the per-round paired differences, not the difference of
        // the two rows above it; the harness carries the rest of the protocol. Every
        // reading here was taken on the machine the stand measures — an Apple M3 Pro,
        // 14-core GPU, 18 GiB, macOS — and `tests/synthesis_front_route.rs` is the
        // artifact behind them. On a quiet machine repeated takes reproduce every arm
        // within half a millisecond and every delta within 0.8 ms; a take taken on a loaded
        // machine reads both arms up by a few per cent and leaves the deltas in the same
        // band — the arms move with the machine, the paired delta does not, which is why
        // the delta is the figure. Every length wins or holds, and the rounds agree
        // everywhere but 26 frames — the one length whose delta sits inside its own spread,
        // and the only one where a take's rounds have disagreed (6 to 9 of 9). The bar the
        // two reference lengths set is that they do not get dearer, and neither does: 42
        // falls by 0.86–1.36 ms and 72 by 1.54–2.29 ms across takes, no round of ours
        // putting either length above its @100 arm — an independent take reads the 72-frame
        // cell as a hold inside its own spread, which is the same bar. So the `pre_conv`
        // route change at *every* window of 72 or fewer costs nothing measurable at either
        // reference. The fourth family is the speaker encoder's first convolution, a
        // unit-height lift at a reference's mel frame count and newly admitted below 100
        // frames: 11.78 → 11.26 ms at 32 frames, 13.91 → 13.49 at 64 and 17.72 → 17.15 at
        // 99, 5 of 5 rounds each, a hold inside the spread at 16 (10.53 → 10.32, 4 of 5),
        // and unchanged at 128 where it was already admitted.
        //
        // The fifth is the recognition conv front at a short one-piece input. An input
        // that fits one piece is compiled at its exact frame count
        // (`transcribe::read_whole` hands `frames_of(len)` to `Clip::once`), so a short
        // one reaches the front below every rung: the third convolution's extent is
        // `16 · asr_tokens(frames)`, under 100 at 48 frames and below, and the second's
        // `32 · t2` — the token count after two of the three — at 12 and below, both
        // clearing the K (4320) and channel (480) floors there. So two convs move at 12
        // frames and below and one at 16 to 48, and every length wins: 3.55 → 1.17 ms at
        // one frame (the noisiest cell — its @100 arm spans 3.3 to 5.5 ms within a take's
        // rounds and its median 3.55 to 5.08 across takes), 4.92 → 1.20 at eight,
        // 7.10 → 1.14 at 12, 2.96 → 1.15 at 16, 3.77 → 1.41 at 32 and 6.05 → 1.77 at 48,
        // 5 of 5 rounds each.
        //
        // The rest of the product does not move, measured as a step census rather than
        // argued: the recognition conv front at the rungs (128 and 768) and at the wake
        // ear's own 76-frame window, and the synthesis stage groups 4..6 and 6..8 at
        // every span the stand's part lengths cut, are step-for-step identical under both
        // floors, and their built arenas identical to the byte (the arena report's size,
        // scratch included). Where a convolution there is not above 100 already it is
        // refused by a non-spatial gate — the ASR front's first conv by the K floor (9
        // terms), the depthwise lifts by their groups, the codec's `post_conv` by the
        // channel floor. The CED tagger holds no convolution at all (its patch embedding
        // is a matmul).
        //
        // Numerics do not move. The harness compares the two arms' outputs on a
        // deterministic ramp — a constant fill cannot show a reordered sum — and finds 0
        // differing elements in every cell. On real inputs two readings agree: the stand
        // re-taken with this default reproduces all six pinned witnesses exactly
        // (recognition's three answer digests, both synthesis digests, the wake score),
        // so nothing was re-pinned; and the newly routed product shapes the stand never
        // reaches — short one-piece recognition inputs — are read on a pinned recording at
        // 12 and 48 frames from 60 s in (the 12-frame cell is the one where the stride-2
        // convolution is newly admitted), both returning the same ids, the same sampled
        // values and the same text under either floor (the 48-frame cell's is "como.",
        // the 12-frame cell's empty). The two routes share an accumulation order on these
        // shapes: evidence, not a guarantee, exactly as for the 800 → 100 move before this
        // one.
        //
        // The memory plan grows only where a panel appears that was not there before.
        // The reservation is a *maximum* over the convs a graph admits, not a sum, so a
        // newly admitted shape can only move it by being the largest panel its graph
        // holds — and from four frames up the front's own 32·frames conv is that panel
        // under either floor, so the built arena (`RLX_WGPU_ARENA_REPORT`'s "plan +
        // scratch -> size", the figure the panel lands in) is identical to the byte:
        // 208.8 MB at eight frames, 226.0 at the shipped 23, 282.2 at 72. Below four
        // frames the old floor refused every conv in the front group, so the panel is new
        // there; each growth below is the difference of its two sizes, both in MB and to
        // the precision shown: 200.0816 → 200.7656 (+0.684) at one frame, 200.5535 →
        // 201.9128 (+1.359) at two and 201.0212 → 203.0600 (+2.039) at three — the largest
        // growth the move has. Two other families book scratch. The speaker encoder's
        // first conv, at 16, 32, 64 and 99 mel frames: 48.6185 → 48.6595 (+0.041),
        // 49.2165 → 49.2984 (+0.082), 50.4125 → 50.5764 (+0.164) and 51.7207 → 51.9741
        // (+0.253), nothing at 128 where it already qualified. The recognition front, at
        // one, eight and twelve frames: 16.8590 → 17.4120 (+0.553), 17.5999 → 18.7058
        // (+1.106) and 18.0934 → 19.7523 (+1.659), and nothing from 16 frames up, where
        // its second conv's panel is the larger one under either floor already. No stand
        // row runs any of these shapes: the front's shortest part is 23 frames, the speaker
        // encoder's embedding is pinned rather than encoded, and the stand's recognition
        // inputs are long enough to cut into pieces at the rungs.
        //
        // Shapes in graphs outside the product are not measured. A convolution of extent
        // under 100 there moves to the materialising route unmeasured, where before this
        // move only the [100, 800) band did: that is the accepted cost of going by the
        // floor rather than by a rule derived from what the two routes do at a given
        // shape.
        //
        // One consequence the move carries rather than implies: execution under an active
        // extent. The accelerated step is not extent-safe and the hint is all-or-nothing
        // per graph, so a graph whose convolution moves to it loses that mode entirely,
        // not just for the moved step. It is not a value difference — the run proceeds at
        // the full compiled extent, with no compile-time refusal, no route change, no
        // host fallback and no wrong result — and by default it is silent: the only
        // signal is one stderr line behind a debug-classified switch whose registry
        // summary describes something else, and the executable's "can honour the hint"
        // capability is a static constant that cannot express the per-graph veto. Nothing
        // on our own shipped path runs in that mode (it is a run-time hint set on an
        // already-compiled executable; only the bucketed-cache helpers set it in-tree,
        // and they have no in-tree caller), so the loss falls on users of that mode.
        //
        // The 1×1 route's own bar is `crate::backend::POINTWISE_MIN_COL_ELEMS`, which
        // this floor does not move, and the device-free route-decision checks derive
        // their fixtures from the effective floors; the one that pins this value is
        // `the_shipped_floor_is_one`.
        Self {
            gdn_host: reg::flag("RLX_WGPU_GDN_HOST"),
            im2col_min_spatial: reg::parse_or("RLX_WGPU_IM2COL_MIN_SPATIAL", 1usize),
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
