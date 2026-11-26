// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The four-bit block decode the row-widening routes call, on the product's own weights:
//! every Q4_K block of both model files against `rlx_gguf::dequant_q4_k_block`, every
//! row of the one-row multiply against `rlx_gguf::q4_k_dot_q8_k`, the route's answer with
//! and without the change, what the decode costs per weight, whether the single-worker arm
//! gains too, and whether the saving reaches the recognition row on the processor.
//!
//! `RLX_GGUF_ROW_DECODE=0` is the change's own arm — each scheme's reference block decode
//! (`rlx_gguf::dequant_q4_k_block`, `dequant_q6_k_block`) through the staging copy — so
//! every reading here is a same-session pair rather than two runs compared across a rebuild.
//!
//! Needs the installed model tree and the pinned corpus. `--test-threads=1` is part of
//! the invocation, not cosmetic: the row take forces `MAHBOT_DEVICE=cpu`, which is
//! process-global and read once at the product's first compile, and `support::solo`
//! refuses a second subject rather than serialising it.
//!
//! ```text
//! cargo test --release -p mahbot-inference --features hand-run --test q4k_decode \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use mahbot_inference::loader::{Class, Model, Role, Store};
use rlx_gguf::{GgmlType, Q4K_BLOCK_BYTES, QK_K};
use rlx_ir::quant::QuantScheme;

mod support;

/// The prefill's row count: a recognition chunk is 79 rows through the body.
const BODY_ROWS: usize = 79;

/// The recognition pair's two files, the body first — a tree without them skips.
fn models() -> Option<Vec<Model>> {
    let (dir, body, sound) = support::pair_paths(Role::Recognition, &[])?;
    Some(
        [body, sound]
            .into_iter()
            .map(|name| {
                let path = dir.join(name);
                Model::open(&path).unwrap_or_else(|error| panic!("{}: {error:#}", path.display()))
            })
            .collect(),
    )
}

/// One packed linear of `ggml`: the file it lives in, its name and its shape.
struct Weight<'a> {
    model: &'a Model,
    name: &'a str,
    k: usize,
    n: usize,
    weights: usize,
}

/// The pair's widest packed linear of `ggml` — the one whose unpack costs most.
fn widest<'a>(models: &'a [Model], ggml: GgmlType) -> Option<Weight<'a>> {
    models
        .iter()
        .flat_map(|model| model.plans.iter().map(move |plan| (model, plan)))
        .filter(|(_, plan)| {
            plan.file == ggml && plan.store == Store::Packed && plan.class == Class::Linear2D
        })
        .max_by_key(|(_, plan)| plan.ggml_shape.iter().product::<usize>())
        .map(|(model, plan)| Weight {
            model,
            name: &plan.name,
            k: plan.ggml_shape[0],
            n: plan.ggml_shape[1],
            weights: plan.ggml_shape.iter().product(),
        })
}

/// `m` activation rows of width `k` — the harness's own fixture, since the route's
/// answer is held against itself and needs no model activations.
fn activation(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i as f32 + 1.0) * 0.013).sin())
        .collect()
}

/// Median of nine passes over `bytes`, as nanoseconds per weight: `decode` widens one
/// `block_bytes`-sized block and writes `written` values, and the sink takes one from each
/// end of that region — inside what the decoder wrote, so nothing above can drop the
/// stores and a scheme whose block is not the super-block is sampled in its own range.
fn decode_rate(
    bytes: &[u8],
    block_bytes: usize,
    weights: f64,
    written: usize,
    decode: fn(&[u8], &mut [f32; QK_K]),
) -> (f64, f32) {
    let mut out = [0f32; QK_K];
    let mut sink = 0f32;
    let mut times = Vec::with_capacity(9);
    for _ in 0..9 {
        let t = Instant::now();
        for b in bytes.chunks_exact(block_bytes) {
            decode(b, &mut out);
            sink += out[0] + out[written - 1];
        }
        times.push(t.elapsed().as_secs_f64() * 1e9);
    }
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2] / weights, sink)
}

/// [`rlx_gguf::q4k_fast::block`] in [`decode_rate`]'s shape.
fn four_bit(block: &[u8], out: &mut [f32; QK_K]) {
    rlx_gguf::q4k_fast::block(block.try_into().expect("one Q4_K block"), out);
}

/// [`rlx_gguf::dequant_q8_0_block`] in [`decode_rate`]'s shape: its block is 32 values,
/// not a super-block.
fn eight_bit(block: &[u8], out: &mut [f32; QK_K]) {
    rlx_gguf::dequant_q8_0_block(block, &mut out[..rlx_gguf::QK8_0]);
}

/// Run one packed multiply with the arm it takes asserted from the counter: `rows` weight
/// rows must be widened, all on the arm the switch names. The two arms write the same values
/// by construction, so the counter is what keeps a take from pricing an inert switch.
fn on_arm<T>(arm: bool, rows: usize, call: impl FnOnce() -> T) -> T {
    let before = rlx_cpu::gguf_matmul::row_decodes();
    let out = call();
    let after = rlx_cpu::gguf_matmul::row_decodes();
    let widened = (after.0 - before.0, after.1 - before.1);
    let want = if arm {
        (rows as u64, 0)
    } else {
        (0, rows as u64)
    };
    assert_eq!(
        widened, want,
        "the switch is inert, or this host's route decodes through dequant_block: rows \
         widened (direct, staged) was {widened:?}, expected {want:?}"
    );
    out
}

/// One route's call measured both ways: `measure` runs with the switch on and off,
/// alternating across `rounds` rounds, and each arm's times come back in round order — a
/// same-session pair rather than two runs compared across a rebuild. `measure` sees the
/// round and the arm, so a take whose noise matters can print the spread as it runs.
fn paired(rounds: usize, mut measure: impl FnMut(usize, bool) -> f64) -> (Vec<f64>, Vec<f64>) {
    let (mut direct, mut staged) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
    for round in 0..rounds {
        let order = if round % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for arm in order {
            let ms = support::with_env("RLX_GGUF_ROW_DECODE", if arm { "1" } else { "0" }, || {
                measure(round, arm)
            });
            if arm {
                direct.push(ms)
            } else {
                staged.push(ms)
            }
        }
    }
    (direct, staged)
}

/// An arm's median, from [`paired`]'s times.
fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// **Every value is the reference's.** Both model files, every packed four-bit block,
/// held against `rlx_gguf::dequant_q4_k_block` value by value — not a tolerance, and
/// not the route's answer, which could hide a difference behind a summation.
#[test]
#[ignore = "needs the installed model tree"]
fn every_real_four_bit_block_is_the_reference_value() {
    let Some(models) = models() else { return };
    let (mut blocks, mut differing, mut worst) = (0usize, 0usize, 0f32);
    let mut want = [0f32; QK_K];
    let mut got = [0f32; QK_K];
    for model in &models {
        for plan in &model.plans {
            if plan.file != GgmlType::Q4K || plan.store != Store::Packed {
                continue;
            }
            model
                .with_mapped(&plan.name, |bytes| {
                    assert_eq!(
                        bytes.len() % Q4K_BLOCK_BYTES,
                        0,
                        "{} is not a whole number of blocks",
                        plan.name
                    );
                    for b in bytes.chunks_exact(Q4K_BLOCK_BYTES) {
                        rlx_gguf::dequant_q4_k_block(b, &mut want);
                        rlx_gguf::q4k_fast::block(b.try_into().expect("one block"), &mut got);
                        blocks += 1;
                        for (g, w) in got.iter().zip(&want) {
                            if g.to_bits() != w.to_bits() {
                                differing += 1;
                                worst = worst.max((g - w).abs());
                            }
                        }
                    }
                })
                .unwrap_or_else(|error| panic!("{}: {error:#}", plan.name));
        }
    }
    println!(
        "four-bit blocks={blocks} weights={} differing={differing} worst={worst:e}",
        blocks * QK_K
    );
    assert!(blocks > 0, "the pair carries no packed four-bit weight");
    assert_eq!(differing, 0, "the decode is not the reference's");
}

/// **Every row dot is the reference's.** Both model files, every packed four-bit linear,
/// every row: the one-row multiply the processor's decode takes, held against a fold of
/// `rlx_gguf::q4_k_dot_q8_k` — the arm it replaces — over the product's own weights, on
/// one Q8_K quantised activation row. Bit for bit, not a tolerance: the block is integer
/// arithmetic up to its final scale, so the vectorised correction has no licence to move
/// one.
///
/// Three ambient switches are pinned, because each one moves either side of this
/// comparison off the integer Q8_K arm: `RLX_DEQUANT_CACHE=0` is the product's own run form
/// and is what routes a single row there at all, `RLX_REPRODUCIBLE` stands every
/// single-row integer route down (`prefer_cached_blas` reads it before the cache), and
/// `RLX_Q4K_EXACT_GEMV` is a different kernel on f32 activations. A shell exporting any of
/// them would report every row as differing — a false number, not a moved bit.
#[test]
#[ignore = "needs the installed model tree"]
fn every_real_four_bit_row_dot_is_the_reference_value() {
    let _alone = support::solo("the four-bit row dot against the reference");
    let Some(models) = models() else { return };
    let (mut rows, mut differing, mut worst) = (0usize, 0usize, 0f32);
    // Pinned for the whole sweep, process-wide as the route reads them.
    fn pinned<T>(body: impl FnOnce() -> T) -> T {
        support::with_env("RLX_REPRODUCIBLE", "0", || {
            support::with_env("RLX_Q4K_EXACT_GEMV", "0", || {
                support::with_env("RLX_DEQUANT_CACHE", "0", body)
            })
        })
    }
    pinned(|| {
        for model in &models {
            for plan in &model.plans {
                if plan.file != GgmlType::Q4K
                    || plan.store != Store::Packed
                    || plan.class != Class::Linear2D
                {
                    continue;
                }
                let (k, n) = (plan.ggml_shape[0], plan.ggml_shape[1]);
                if !k.is_multiple_of(QK_K) || n == 0 {
                    continue;
                }
                let x = activation(1, k);
                let mut x_q8 = vec![0u8; k / QK_K * rlx_gguf::Q8K_BLOCK_BYTES];
                rlx_gguf::quantize_q8_k_row(&x, &mut x_q8);
                let mut got = vec![0f32; n];
                model
                    .with_mapped(&plan.name, |bytes| {
                        rlx_cpu::gguf_matmul::gguf_matmul_bt_dispatch_at(
                            &x,
                            bytes,
                            &mut got,
                            1,
                            k,
                            n,
                            QuantScheme::GgufQ4K,
                            0,
                        );
                        let row_bytes = k / QK_K * Q4K_BLOCK_BYTES;
                        for (j, g) in got.iter().enumerate() {
                            let row = &bytes[j * row_bytes..(j + 1) * row_bytes];
                            let want: f32 = row
                                .chunks_exact(Q4K_BLOCK_BYTES)
                                .zip(x_q8.chunks_exact(rlx_gguf::Q8K_BLOCK_BYTES))
                                .fold(0.0f32, |acc, (q4, q8)| {
                                    acc + rlx_gguf::q4_k_dot_q8_k(q4, q8)
                                });
                            rows += 1;
                            if g.to_bits() != want.to_bits() {
                                differing += 1;
                                worst = worst.max((g - want).abs());
                            }
                        }
                    })
                    .unwrap_or_else(|error| panic!("{}: {error:#}", plan.name));
            }
        }
    });
    println!("four-bit rows={rows} differing={differing} worst={worst:e}");
    assert!(rows > 0, "the pair carries no packed four-bit linear");
    assert_eq!(differing, 0, "the row dot is not the reference's");
}

/// **The route answers the same either way**, on the widest real weight of each scheme
/// the pair's files carry — the four-bit one the decode is about and the six-bit one that
/// shares the direct write. The eight-bit block is not the super-block, so the caller's
/// `block_elems != QK_K` guard leaves it on the staging path, where there is no second arm
/// to compare.
#[test]
#[ignore = "needs the installed model tree"]
fn the_route_answers_the_same_with_and_without_the_change() {
    let _alone = support::solo("the route's answer with and without the change");
    let Some(models) = models() else { return };
    for (ggml, scheme, m) in [
        (GgmlType::Q4K, QuantScheme::GgufQ4K, BODY_ROWS),
        (GgmlType::Q6K, QuantScheme::GgufQ6K, BODY_ROWS),
    ] {
        let Some(w) = widest(&models, ggml) else {
            println!("skip: the pair carries no packed {ggml:?} linear");
            continue;
        };
        let x = activation(m, w.k);
        let mut on = vec![0f32; m * w.n];
        let mut off = vec![0f32; m * w.n];
        w.model
            .with_mapped(w.name, |bytes| {
                on_arm(true, w.n, || {
                    support::with_env("RLX_GGUF_ROW_DECODE", "1", || {
                        rlx_cpu::gguf_matmul::gguf_matmul_bt(
                            &x, bytes, &mut on, m, w.k, w.n, scheme,
                        )
                    })
                });
                on_arm(false, w.n, || {
                    support::with_env("RLX_GGUF_ROW_DECODE", "0", || {
                        rlx_cpu::gguf_matmul::gguf_matmul_bt(
                            &x, bytes, &mut off, m, w.k, w.n, scheme,
                        )
                    })
                });
            })
            .unwrap_or_else(|error| panic!("{}: {error:#}", w.name));
        let (differing, worst) = support::output_difference(&[on], &[off]);
        println!(
            "{ggml:?} {} m={m} k={} n={}: differing={differing} worst={worst:e}",
            w.name, w.k, w.n
        );
        assert_eq!(differing, 0, "the route's answer moved for {ggml:?}");
    }
}

/// **What the decode costs per weight.** The reference decode and the four-bit
/// arrangement over one real four-bit weight, and each scheme the change does not
/// re-arrange over its own widest weight with its own decoder and its own block stride —
/// so a take says whether those schemes are where they were.
#[test]
#[ignore = "needs the installed model tree"]
fn the_decode_rate() {
    let Some(models) = models() else { return };
    let Some(w) = widest(&models, GgmlType::Q4K) else {
        println!("skip: the pair carries no packed four-bit linear");
        return;
    };
    w.model
        .with_mapped(w.name, |bytes| {
            let weights = w.weights as f64;
            let (reference, sink_a) = decode_rate(
                bytes,
                Q4K_BLOCK_BYTES,
                weights,
                QK_K,
                rlx_gguf::dequant_q4_k_block,
            );
            let (arrangement, sink_b) =
                decode_rate(bytes, Q4K_BLOCK_BYTES, weights, QK_K, four_bit);
            println!(
                "{} weights={} blocks={}: reference {reference:.3} ns/weight ({:.2} G w/s), \
                 four-bit arrangement {arrangement:.3} ns/weight ({:.2} G w/s), {:.2}x",
                w.name,
                w.weights,
                bytes.len() / Q4K_BLOCK_BYTES,
                1.0 / reference,
                1.0 / arrangement,
                reference / arrangement,
            );
            assert!(sink_a + sink_b != 0.0, "the passes wrote nothing");
        })
        .unwrap_or_else(|error| panic!("{}: {error:#}", w.name));
    for (ggml, scheme, decode) in [
        (
            GgmlType::Q6K,
            QuantScheme::GgufQ6K,
            rlx_gguf::dequant_q6_k_block as fn(&[u8], &mut [f32; QK_K]),
        ),
        (GgmlType::Q8_0, QuantScheme::GgufQ8_0, eight_bit),
    ] {
        let Some(other) = widest(&models, ggml) else {
            continue;
        };
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let written = scheme.gguf_block_size() as usize;
        other
            .model
            .with_mapped(other.name, |bytes| {
                let weights = other.weights as f64;
                let (rate, sink) = decode_rate(bytes, block_bytes, weights, written, decode);
                println!(
                    "{ggml:?} {} weights={} blocks={}: reference {rate:.3} ns/weight ({:.2} G w/s)",
                    other.name,
                    other.weights,
                    bytes.len() / block_bytes,
                    1.0 / rate
                );
                assert!(sink != 0.0, "the passes wrote nothing");
            })
            .unwrap_or_else(|error| panic!("{}: {error:#}", other.name));
    }
}

/// **What a prefill pass pays.** Every packed linear a prompt chunk multiplies — the
/// four-bit weights the decode arrangement is about and the six-bit ones that only lose
/// the staging copy — through the route with the switch on and off, alternating and
/// taking the median, each call's arm asserted from the counter. The output head is left
/// out: a chunk multiplies it by its last row alone, so it is not part of a pass's unpack.
/// No row percentage is printed: that needs a pass count and a row time from a record file,
/// and a re-taken record moves both under it.
#[test]
#[ignore = "needs the installed model tree"]
fn the_route_marginal_per_pass() {
    let _alone = support::solo("the prefill pass's route marginal");
    let Some(models) = models() else { return };
    let arms = [
        (GgmlType::Q4K, QuantScheme::GgufQ4K),
        (GgmlType::Q6K, QuantScheme::GgufQ6K),
    ];
    let mut weights = [0usize; 2];
    let mut direct_ms = [0f64; 2];
    let mut staged_ms = [0f64; 2];
    for model in &models {
        for plan in &model.plans {
            if plan.store != Store::Packed
                || plan.class != Class::Linear2D
                || plan.name == "output.weight"
            {
                continue;
            }
            let Some(arm) = arms.iter().position(|(ggml, _)| *ggml == plan.file) else {
                continue;
            };
            let scheme = arms[arm].1;
            let (k, n) = plan.in_out();
            let x = activation(BODY_ROWS, k);
            let mut out = vec![0f32; BODY_ROWS * n];
            let (d, s) = model
                .with_mapped(&plan.name, |bytes| {
                    let (direct, staged) = paired(3, |_, on| {
                        on_arm(on, n, || {
                            let t = Instant::now();
                            rlx_cpu::gguf_matmul::gguf_matmul_bt(
                                &x, bytes, &mut out, BODY_ROWS, k, n, scheme,
                            );
                            t.elapsed().as_secs_f64() * 1e3
                        })
                    });
                    (median(direct), median(staged))
                })
                .unwrap_or_else(|error| panic!("{}: {error:#}", plan.name));
            println!(
                "{:?} {} m={BODY_ROWS} k={k} n={n}: staged {s:8.3} ms  direct {d:8.3} ms  \
                 marginal {:6.3} ms over {} weights",
                plan.file,
                plan.name,
                s - d,
                k * n
            );
            weights[arm] += k * n;
            direct_ms[arm] += d;
            staged_ms[arm] += s;
        }
    }
    for (arm, label) in ["four-bit", "six-bit"].iter().enumerate() {
        println!(
            "{label}: {} weights, route {:.1} ms staged against {:.1} ms direct, marginal \
             {:.3} ms a pass ({:.1} % of the route)",
            weights[arm],
            staged_ms[arm],
            direct_ms[arm],
            staged_ms[arm] - direct_ms[arm],
            100.0 * (staged_ms[arm] - direct_ms[arm]) / staged_ms[arm],
        );
    }
    let all = staged_ms[0] + staged_ms[1] - direct_ms[0] - direct_ms[1];
    println!(
        "one prompt prefill pass: {} weights, {:.1} ms staged against {:.1} ms direct, \
         marginal {all:.3} ms",
        weights[0] + weights[1],
        staged_ms[0] + staged_ms[1],
        direct_ms[0] + direct_ms[1],
    );
}

/// **The single-worker arm gains too.** `gguf_matmul_bt_serial` is the row-dot route: one
/// weight row widened once, then dotted against every activation row, no tile and no
/// vendor BLAS. A one-worker build takes the same route, and it takes the direct
/// arrangement by default as well, so its blast radius is measured rather than assumed —
/// four activation rows, the widest weight of each scheme the change touches, the switch
/// on and off and alternating, the two answers held bit-identical. The other no-BLAS arm,
/// `gguf_matmul_bt_rows_generic_gemm`, decodes through `dequant_block` and is untouched.
#[test]
#[ignore = "needs the installed model tree"]
fn the_single_worker_arm_gains_too() {
    let _alone = support::solo("the single-worker arm's marginal");
    let Some(models) = models() else { return };
    const ROWS: usize = 4;
    for (ggml, scheme) in [
        (GgmlType::Q4K, QuantScheme::GgufQ4K),
        (GgmlType::Q6K, QuantScheme::GgufQ6K),
    ] {
        let Some(w) = widest(&models, ggml) else {
            continue;
        };
        let x = activation(ROWS, w.k);
        let mut on = vec![0f32; ROWS * w.n];
        let mut off = vec![0f32; ROWS * w.n];
        let (d, s) = w
            .model
            .with_mapped(w.name, |bytes| {
                let (direct, staged) = paired(3, |_, arm| {
                    let out = if arm { &mut on } else { &mut off };
                    on_arm(arm, w.n, || {
                        let t = Instant::now();
                        rlx_cpu::gguf_matmul::gguf_matmul_bt_serial(
                            &x, bytes, out, ROWS, w.k, w.n, scheme,
                        );
                        t.elapsed().as_secs_f64() * 1e3
                    })
                });
                (median(direct), median(staged))
            })
            .unwrap_or_else(|error| panic!("{}: {error:#}", w.name));
        let (differing, worst) = support::output_difference(&[on], &[off]);
        println!(
            "{ggml:?} {} m={ROWS} k={} n={} row-dot arm: staged {s:.3} ms, direct {d:.3} ms, \
             marginal {:+.3} ms ({:+.1} %), differing={differing} worst={worst:e}",
            w.name,
            w.k,
            w.n,
            s - d,
            100.0 * (s - d) / s,
        );
        assert_eq!(differing, 0, "the row-dot arm's answer moved for {ggml:?}");
    }
}

/// **The recognition row on the processor.** The stand's long recording, both arms
/// alternating in one session: the whole call's wall time, the product's own phase line
/// beside it (printed by the call, in call order), and the answers held equal — each
/// reading checked to tile the input and carry words first, so two refusals cannot agree
/// vacuously, and the decode moves no value, so a row that read different text would be a
/// defect rather than a cost. The prefill column is the instrument — the decode column
/// beside it is a hundred times the effect and moves by more than the effect between
/// rounds — but on a loaded box even the prefill column's spread can exceed the effect,
/// so read it beside [`the_route_marginal_per_pass`], which resolves one pass and repeats.
#[test]
#[ignore = "needs the installed model tree and the pinned corpus; run by hand"]
fn the_recognition_row_on_the_processor() {
    let _alone = support::force_processor("the four-bit decode's row take");
    // SAFETY: `force_processor` holds the binary's one subject, so nothing else is reading
    // the environment; the product's first compile is below.
    unsafe { std::env::set_var("MAHBOT_PHASE_SPLIT", "1") };
    let Some(mut row) = support::Row::open(None) else {
        return;
    };
    let call = |row: &mut support::Row, direct: bool| -> f64 {
        support::with_env(
            "RLX_GGUF_ROW_DECODE",
            if direct { "1" } else { "0" },
            || {
                let t = Instant::now();
                row.call();
                t.elapsed().as_secs_f64() * 1e3
            },
        )
    };
    // One warm-up each, so the model open and the graph build fall outside the take.
    let warm_direct = call(&mut row, true);
    support::assert_settled_on_the_processor();
    let warm_staged = call(&mut row, false);
    println!("warm-up: direct {warm_direct:.0} ms, staged {warm_staged:.0} ms");
    let (direct, staged) = paired(4, |round, arm| {
        let ms = call(&mut row, arm);
        println!(
            "round {round}: {} {ms:.0} ms",
            if arm { "direct" } else { "staged" }
        );
        ms
    });
    let deltas: Vec<f64> = direct.iter().zip(&staged).map(|(d, s)| s - d).collect();
    let delta = median(deltas.clone());
    let (d, s) = (median(direct), median(staged));
    println!(
        "recognition 318s on the processor: staged {s:.0} ms, direct {d:.0} ms; per-round \
         difference {deltas:.0?}, median {delta:.0} ms ({:.2} % of the call) — the prefill \
         column of each call's phase line is the instrument, the call is not",
        100.0 * delta / s,
    );
}
