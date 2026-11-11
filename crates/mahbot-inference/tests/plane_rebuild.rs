// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A resident plane cache moved from one compiled decode graph into the graph that
//! replaces it — the bucket rebuild a walk or a render takes when its live rows
//! reach the compiled capacity. The rebuild reads the old graph's slots into one
//! host buffer and writes the new graph's, so what it must carry is the live rows
//! and nothing else: every layer's K and V plane keeps the rows it had, and the
//! grown graph's rows above them stay zero.
//!
//! Needs the installed model directory. Run with
//! `cargo test --release -p mahbot-inference --features hand-run --test plane_rebuild
//! -- --ignored --nocapture`.

use std::sync::Arc;

use mahbot_inference::graphs::{self, CACHE_BUCKET};
use mahbot_inference::loader::{Pair, Role};
use rlx_runtime::Device;

mod support;

/// A pair, or a printed `skip:` when the installed tree has no models.
fn pair() -> Option<Arc<Pair>> {
    match Pair::of_role(Role::Recognition) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("skip: {e}");
            None
        }
    }
}

/// `layers` planes of `rows · hkv` values, distinct per layer so a plane written into
/// its neighbour's slot cannot pass.
fn seeded(layers: usize, rows: usize) -> Vec<f32> {
    (0..layers)
        .flat_map(|l| {
            support::ramped_row(rows)
                .into_iter()
                .map(move |v| v + l as f32)
        })
        .collect()
}

#[test]
#[ignore = "needs the installed model directory"]
fn a_rebuilt_cache_carries_the_live_rows_and_nothing_above_them() {
    let Some(pair) = pair() else { return };
    let cfg = pair.body.config().clone();
    let hkv = cfg.kv_heads * cfg.head_dim;
    const CAPACITY: usize = 8;
    const LIVE: usize = 5;
    let grown = CAPACITY + CACHE_BUCKET;
    let k = seeded(cfg.layers, LIVE * hkv);
    let v = seeded(cfg.layers, LIVE * hkv);

    let build = |capacity: usize| -> graphs::Compiled {
        let built = graphs::body_capacity_chain_planes(pair.body.clone(), capacity)
            .expect("build the plane-cache decode body");
        graphs::Compiled::new(Device::Gpu, built).expect("compile the plane-cache body")
    };
    let mut from = build(CAPACITY);
    let mask = from
        .runner
        .resident_cache_planes(cfg.layers, CAPACITY, LIVE, hkv, &k, &v)
        .expect("the seed rows land in the graph's own slots");
    assert_eq!(
        mask,
        graphs::resident_mask(LIVE, CAPACITY),
        "the seed answers the mask of the rows it wrote"
    );
    let rows = from
        .runner
        .read_plane_rows(cfg.layers, hkv, LIVE)
        .expect("the rebuild reads every layer's live rows");
    let mut to = build(grown);
    to.runner
        .write_plane_rows(cfg.layers, &rows)
        .expect("the rebuild writes every layer's live rows");

    // One read past the live rows answers both halves of the claim: the rows the seed
    // wrote are what the rebuild carried, and the first row above them is the zero the
    // device wrote, not a copy of a row the old graph never had.
    let read = |graph: &graphs::Compiled| {
        graph
            .runner
            .read_plane_rows(cfg.layers, hkv, LIVE + 1)
            .expect("every layer's planes are readable")
    };
    let seeded_planes = read(&from);
    let rebuilt_planes = read(&to);
    for (l, (seeded, rebuilt)) in seeded_planes
        .chunks_exact(2)
        .zip(rebuilt_planes.chunks_exact(2))
        .enumerate()
    {
        for (name, src, rows) in [("k", &k, &seeded[0]), ("v", &v, &seeded[1])] {
            assert_eq!(
                &rows[..LIVE * hkv],
                &src[l * LIVE * hkv..(l + 1) * LIVE * hkv],
                "{name}{l} holds the rows the seed wrote"
            );
        }
        for (name, src, rows) in [("k", &k, &rebuilt[0]), ("v", &v, &rebuilt[1])] {
            assert_eq!(
                &rows[..LIVE * hkv],
                &src[l * LIVE * hkv..(l + 1) * LIVE * hkv],
                "{name}{l} kept its live rows through the rebuild"
            );
            assert!(
                rows[LIVE * hkv..].iter().all(|v| *v == 0.0),
                "{name}{l} is not zero above the live rows: {:?}",
                &rows[LIVE * hkv..]
            );
        }
    }
    eprintln!(
        "rebuilt {CAPACITY} → {grown} rows over {} layers at hkv {hkv}",
        cfg.layers
    );
}
