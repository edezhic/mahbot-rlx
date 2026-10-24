// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The embedding cells end to end, against the reference engine: the
//! tokenizer's ids and the graph's 768-wide vector, on the pinned GGUF and the
//! pinned article.
//!
//! **What it pins.** Two things, in this order:
//!
//! * **the ids, exactly.** The whole article and both cell texts are tokenised
//!   and the id sequence's SHA-256 (over the ids as little-endian `u32`s) is
//!   compared with the reference engine's own output — `llama-tokenize` from the
//!   build `tests/fixtures/embeddinggemma2-golden.json` names in its
//!   `engine_version` (llama.cpp `b11724`, commit
//!   `99db719bdf07f7d537431ca556a0512bf3d995b6`), which is also what wrote that
//!   fixture. A hash is used rather than an
//!   11 567-entry table because the comparison it stands for is token for
//!   token: one id different anywhere changes it.
//! * **the vector, to a cosine and a per-element bar.** Each cell's graph
//!   output, L2-normalised, is compared with
//!   `tests/fixtures/embeddinggemma2-golden.json` — the same engine's
//!   `llama-embedding --pooling mean --embd-normalize 2` output, cut from the
//!   same two texts. Both comparisons are asserted: the cosine against
//!   **0.99995** and the largest per-element difference against **3.0e-4**.
//!   The second bar is about 2.3× the worst figure measured over both routes
//!   and both lengths (1.32e-4), so it admits the fixture's accumulation
//!   order's residue and not a regression.
//!
//! **The tolerance is the measured one.** There are two routes and the test
//! prints which one it ran as `gemm=`: `f16coop` (the packed weights
//! dequantised to f16 on the host, multiplied by `Op::MatMul`, served by the
//! cooperative f16 kernel with an f32 accumulator — the default where the device
//! admits it) and `packed` (the file's Q8_0 bytes through `Op::DequantMatMul`,
//! the portable route and the fallback). `MAHBOT_EMBED_GEMM=packed` asks for the
//! second. Measured on the pinned device and models, a re-take on 2026-10-24
//! over both routes and both cells — the cosine and max-abs are the gate's
//! subject and the only figures it keeps:
//!
//! | cell | route | cosine | max-abs |
//! |---|---|---|---|
//! | 256 | `f16coop` | 1.000000 | 1.02e-4 |
//! | 256 | `packed` | 0.999999 | 1.32e-4 |
//! | 8 192 | `f16coop` | 1.000000 | 8.82e-5 |
//! | 8 192 | `packed` | 1.000000 | 8.57e-5 |
//!
//! So f16 operands with an f32 accumulator do not cost this cell a measurable
//! cosine at either length, and at 256 rows they land marginally closer to the
//! fixture than the packed route does. The same holds for the attention half:
//! the 8 192-row cell's four dense layers multiply their scores in f16 with an
//! f32 accumulator (`backend/attention_route.rs`, the cooperative chunked form)
//! while the 256-row cell is below that form's row floor and inside the band's
//! scratch cap, so its layers take the many-row decomposed f32 arrangement, and
//! the two cells reach the fixture by different arithmetic and both land inside
//! it. The bar's floor is the fixture's own accumulation order, which no route
//! of this graph reproduces exactly; the
//! table above is the tolerance's justification. What the bar no longer
//! tolerates is a drift the old 0.9990 floor would have passed: a multiply that
//! widens its operands, reorders a reduction or drops a mask now shows up in
//! the cosine or in one element rather than only in the printed fourth decimal.
//! What the *figures* above establish is the tolerance; what they do not is a
//! claim about another device — the f16 route is only reachable where
//! `coop_probe::f16_f32acc_path_ok()` passes, and a device that fails it keeps
//! the packed route and the packed route's numbers.
//!
//! **What it does not establish.** Nothing about timing: the run is one cold
//! and five warm passes under whatever load the machine has, while
//! `MEASUREMENTS.md`'s figures are medians of three stand runs on the pinned
//! device. A ×1.3 gate over those figures would fire on a busy machine for
//! reasons that have nothing to do with this code, so the timings are printed
//! and not asserted.
//!
//! **When it runs.** Each test ends in `_with_models` and needs
//! `MAHBOT_MODELS_DIR` (with the pinned GGUF), `MAHBOT_EMBED_TEXT` (with the
//! pinned article) and a wgpu adapter; without them it prints why and returns
//! instead of failing. An input that is *set* and is not what it names is not an
//! absent one: the run fails, because a gate that skipped there would report
//! green having checked nothing. Both cells run the encoder graph: the twenty
//! sliding layers state the model's two-sided ±512 band where it restricts, and
//! the four global layers are dense and state no mask. Which kernel serves either
//! statement is the backend router's choice, not this test's.

use mahbot_inference::graphs;
use mahbot_inference::loader::{self, Model};
use mahbot_inference::reference;
use mahbot_inference::tokenizer::Tokenizer;
use rlx_runtime::Device;
use sha2::{Digest, Sha256};

/// The golden fixture, written by the reference engine — see the header.
const GOLDEN: &str = include_str!("fixtures/embeddinggemma2-golden.json");

/// The gate: the measured agreement is 1.000000 with a worst per-element
/// residue of 1.32e-4 (the header's table), so the cosine bar sits just below
/// the lowest measured figure (0.999999) and the max-abs bar about 2.3× the
/// worst one.
const COSINE_BAR: f32 = 0.99995;
const MAX_ABS_BAR: f32 = 3.0e-4;

/// What the ids' SHA-256 is taken over: the ids as little-endian `u32`s, in
/// order. The same rule `MEASUREMENTS.md`'s `digest` column is not: that one is
/// FNV over the f32 output.
fn ids_sha256(ids: &[usize]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update((*id as u32).to_le_bytes());
    }
    let digest = h.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The reference engine's id digests, from `llama-tokenize` on the pinned GGUF:
/// the whole article (11 567 ids, BOS and EOS included) and the two cell texts
/// (256 and 8 192).
const ARTICLE_IDS_SHA256: &str = "8175f16b0fe5e9a66ecb4762ad139059d3450037307b57cd99ec265899aec4c0";
const CELL_256_IDS_SHA256: &str =
    "7bcd6a83c990bc28e458a5dd1a9fd94daacdbf541e125d1a2fc11f49b7cc6275";
const CELL_8192_IDS_SHA256: &str =
    "7044583d86bcf701d32724e477c9db16422537d2b76156411921c45674c034db";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn patch(golden: &serde_json::Value, rows: usize) -> Vec<f32> {
    golden["cells"][rows.to_string()]["values"]
        .as_array()
        .expect("the fixture holds the cell's values")
        .iter()
        .map(|v| v.as_f64().expect("a float") as f32)
        .collect()
}

/// Whether the run was pointed at one of its inputs explicitly: a variable that
/// is set and names something this gate cannot use is a mis-provisioned run, not
/// an absent one.
fn pinned(var: &str) -> bool {
    std::env::var_os(var).is_some()
}

/// The model, the tokenizer and the article — or the reason this run cannot
/// happen, printed once and returned as `None`.
fn setup() -> Option<(Model, Tokenizer, String)> {
    let dir = match loader::models_dir() {
        Ok(dir) => dir,
        Err(error) => {
            assert!(
                !pinned(loader::MODELS_DIR_ENV),
                "{} is set: {error}",
                loader::MODELS_DIR_ENV
            );
            println!("skipped: {error}");
            return None;
        }
    };
    let path = dir.join(loader::EMBED_MODEL);
    if !path.is_file() {
        assert!(
            !pinned(loader::MODELS_DIR_ENV),
            "{} is set and {} is not on this machine",
            loader::MODELS_DIR_ENV,
            path.display()
        );
        println!("skipped: {} is not on this machine", path.display());
        return None;
    }
    let article = match reference::article() {
        Ok(article) => article,
        // The article is pinned by identity, so a file that is not the pinned
        // extract is a mis-provisioned run: a gate that skipped it would report
        // green having checked nothing it names.
        Err(error) => {
            assert!(
                !pinned(loader::EMBED_TEXT_ENV),
                "{} is set and is not the pinned extract: {error}",
                loader::EMBED_TEXT_ENV
            );
            println!("skipped: {error}");
            return None;
        }
    };
    let model = Model::open(&path).expect("the pinned GGUF opens");
    let tokenizer = Tokenizer::from_model(&model).expect("the pinned GGUF tokenises");
    Some((model, tokenizer, article))
}

/// The ids, token for token: the article and both cell texts against the
/// reference engine's own sequences.
#[test]
fn the_pinned_texts_tokenise_to_the_reference_engine_s_ids_with_models() {
    let Some((_, tokenizer, article)) = setup() else {
        return;
    };
    let got = ids_sha256(&tokenizer.encode_with_specials(&article));
    println!(
        "article ids={} sha256={got}",
        tokenizer.encode(&article).len() + 2
    );
    assert_eq!(got, ARTICLE_IDS_SHA256, "the article's ids moved");

    for (rows, want, content) in [
        (256usize, CELL_256_IDS_SHA256, 254usize),
        (8_192, CELL_8192_IDS_SHA256, 8_190),
    ] {
        let text = reference::cell_text(&article, rows).expect("the pinned cell text");
        let ids = tokenizer.encode_with_specials(&text);
        assert_eq!(ids.len(), content + 2, "cell {rows}: BOS + content + EOS");
        let got = ids_sha256(&ids);
        println!("cell {rows} ids={} sha256={got}", ids.len());
        assert_eq!(got, want, "cell {rows}: the ids moved");
    }
}

/// One cell through the graph, against the fixture: `(cosine, max-abs)`.
fn cell(rows: usize) -> Option<(f32, f32)> {
    let (mut model, tokenizer, article) = setup()?;
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return None;
    }
    // The same route the stand's cell takes, decided the same way: f16 when this
    // device's cooperative-kernel probe passes and every multiply admits its
    // tile, packed otherwise. `MAHBOT_EMBED_GEMM=packed` asks for the packed arm.
    let gemm = model.f16_linears(rows);
    let model = std::sync::Arc::new(model);
    let golden: serde_json::Value = serde_json::from_str(GOLDEN).expect("the fixture parses");
    let want = patch(&golden, rows);
    let text = reference::cell_text(&article, rows).expect("the pinned cell text");
    let ids = tokenizer.encode_with_specials(&text);
    let cfg = model.gemma().clone();
    let picks: Vec<(usize, usize)> = ids.iter().map(|&id| (0, id)).collect();
    let mut x = Vec::with_capacity(rows * cfg.d);
    for row in model
        .table_rows_f32("token_embd.weight", &picks, cfg.d)
        .expect("the token table's rows")
    {
        x.extend_from_slice(&row);
    }
    let mut compiled = graphs::compile_encoder(
        Device::Gpu,
        graphs::encoder(model.clone(), rows).expect("the graph"),
        gemm,
    )
    .expect("the graph compiles");
    let o = compiled.run(&[("x".into(), x)], 5);
    let raw = &o.outs[0];
    assert_eq!(raw.len(), cfg.out_width, "the graph's own output width");
    let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    let normalised: Vec<f32> = raw.iter().map(|v| v / norm).collect();
    let cos = cosine(&normalised, &want);
    let maxabs = normalised
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "cell {rows} tokens={} gemm={} norm={norm:.6} cosine={cos:.6} max_abs={maxabs:.6e} \
         cold_ms={:.2} warm_ms={:.2} build_ms={:.2}",
        ids.len(),
        gemm.token(),
        o.first_ms,
        o.step_ms,
        compiled.compile_ms + compiled.upload_ms,
    );
    Some((cos, maxabs))
}

/// The 256-row cell: the mean of all 256 rows, projected, L2-normalised,
/// against the reference engine.
#[test]
fn the_256_row_cell_matches_the_reference_vector_with_models() {
    let Some((cos, maxabs)) = cell(256) else {
        return;
    };
    assert!(
        cos >= COSINE_BAR,
        "cosine {cos:.6} is below the {COSINE_BAR} bar"
    );
    assert!(
        maxabs <= MAX_ABS_BAR,
        "max-abs {maxabs:.3e} is above the {MAX_ABS_BAR:.1e} bar"
    );
}

/// The 8 192-row cell: the mean of all 8 192 rows, projected, L2-normalised,
/// against the reference engine.
#[test]
fn the_8192_row_cell_matches_the_reference_vector_with_models() {
    let Some((cos, maxabs)) = cell(8_192) else {
        return;
    };
    assert!(
        cos >= COSINE_BAR,
        "cosine {cos:.6} is below the {COSINE_BAR} bar"
    );
    assert!(
        maxabs <= MAX_ABS_BAR,
        "max-abs {maxabs:.3e} is above the {MAX_ABS_BAR:.1e} bar"
    );
}
