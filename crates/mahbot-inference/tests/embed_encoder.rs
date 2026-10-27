// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The embedding cells' decisions that need neither a model file nor a device:
//! the band-vs-no-mask equivalence, the article cut's arithmetic, and the
//! loader's refusal of a `gemma-embedding2` file whose per-layer arrays and
//! tensors disagree.
//!
//! What it pins, and why each is a *decision* rather than an implementation
//! detail:
//!
//! * **a band is only stated where it restricts** — [`graphs::window_mask`]
//!   answers `None` for a sequence the band already covers, so the twenty
//!   sliding layers state the two-sided ±512 mask only at 8 192 rows. The four
//!   global layers are dense and state `MaskKind::None` at every length; which
//!   kernel serves either statement is the backend router's choice. The
//!   equivalence is checked by enumerating the pairs, not by trusting the
//!   boundary;
//! * **the cells' row arithmetic** — BOS, content, EOS, nothing else;
//! * **the article cut** — space-delimited words, from the article's start for
//!   `embed.8192` and from the pinned anchor sentence for `embed.256`, with the
//!   pins refusing any text that is not the one the reference engine was run on;
//! * **the loader's refusals** — a `gemma-embedding2` file whose head-count
//!   array, layer-type pattern, value width or projection shapes disagree with
//!   what the graph builds is refused by name, because the alternative is a
//!   graph that reshapes one width as another. So is a metadata key whose value
//!   is the wrong type, an RMS epsilon that is not positive and finite, and a
//!   vocabulary whose rule set this crate does not implement.
//!
//! What it does not establish: that the encoder's numbers match the reference —
//! that is `tests/embed_model.rs`, which needs the GGUF, the article and a
//! wgpu adapter.

use std::io::Cursor;

use mahbot_inference::graphs;
use mahbot_inference::loader::Model;
use mahbot_inference::reference;
use mahbot_inference::tokenizer::Tokenizer;
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};
use rlx_ir::op::MaskKind;

/// The band's radius the pinned file states (`sliding_window = 1024`, halved by
/// llama.cpp's `LLAMA_SWA_TYPE_SYMMETRIC`).
const RADIUS: usize = 512;

/// Every key a query can reach under a two-sided band of `radius`, over `rows`.
fn band_reaches_all(rows: usize, radius: usize) -> bool {
    (0..rows).all(|q| (0..rows).all(|k| q.abs_diff(k) <= radius))
}

#[test]
fn the_band_is_stated_only_where_it_restricts() {
    // The boundary is `radius + 1` rows: the farthest pair is `(0, rows - 1)`.
    assert!(band_reaches_all(RADIUS + 1, RADIUS));
    assert!(!band_reaches_all(RADIUS + 2, RADIUS));
    assert_eq!(graphs::window_mask(RADIUS + 1, RADIUS), MaskKind::None);
    assert_eq!(
        graphs::window_mask(RADIUS + 2, RADIUS),
        MaskKind::Window(RADIUS)
    );
    // The two cells: 256 needs no mask, 8 192 does.
    assert_eq!(graphs::window_mask(256, RADIUS), MaskKind::None);
    assert_eq!(graphs::window_mask(8_192, RADIUS), MaskKind::Window(RADIUS));
}

#[test]
fn a_cell_is_bos_content_and_eos() {
    assert_eq!(reference::content_tokens(256), 254);
    assert_eq!(reference::content_tokens(8_192), 8_190);
    // Both cells are in the crate's list, and each is its own row.
    assert_eq!(reference::EMBED_CELLS, [256, 8_192]);
}

#[test]
fn the_cut_is_space_delimited_words() {
    let text = "alpha beta\ngamma delta epsilon";
    assert_eq!(reference::words_prefix(text, 1), "alpha");
    assert_eq!(reference::words_prefix(text, 2), "alpha beta\ngamma");
    assert_eq!(reference::words_prefix(text, 4), text);
    // Fewer words than asked for is the whole text, not an error: the cells'
    // counts are checked against their pins, not here.
    assert_eq!(reference::words_prefix(text, 99), text);
}

/// The extract's pins are the reference engine's input: a text that is not the
/// pinned one is refused, and the message names the cell.
#[test]
fn an_unpinned_article_is_refused_by_the_cell_cut() {
    // The anchor is the 256-cell's start, so its absence is its own refusal.
    let err = reference::cell_text("no anchor here at all", 256).expect_err("no anchor");
    assert!(
        format!("{err}").contains(reference::CELL_256_ANCHOR),
        "{err}"
    );
    // With the anchor but the wrong text, the byte count and the sha refuse.
    let mut fake = String::from(reference::CELL_256_ANCHOR);
    fake.push_str(" and then some words that are not the pinned excerpt");
    let err = reference::cell_text(&fake, 256).expect_err("not the pinned excerpt");
    assert!(format!("{err}").contains("the pin says"), "{err}");
    // A row count that is not a cell is named as such.
    let err = reference::cell_text("anything", 300).expect_err("not a cell");
    assert!(format!("{err}").contains("no embedding cell"), "{err}");
}

// ── a synthetic `gemma-embedding2` file ────────────────────────────

/// The smallest shapes the graph's own layer widths can be built at, so the
/// file is a few kilobytes and still exercises every comparison the loader
/// makes: 2 layers, 2 heads of width 4, one KV head, a PLE width of 8.
fn synthetic(overrides: &[(&str, MetaValue)], mutate: &dyn Fn(&mut GgufWriter)) -> GgufWriter {
    let mut w = GgufWriter::new();
    w.set_arch("gemma-embedding2");
    let m = |w: &mut GgufWriter, key: &str, v: MetaValue| w.set_meta(key.to_string(), v);
    m(&mut w, "gemma-embedding2.block_count", MetaValue::U32(2));
    m(
        &mut w,
        "gemma-embedding2.embedding_length",
        MetaValue::U32(8),
    );
    m(
        &mut w,
        "gemma-embedding2.feed_forward_length",
        MetaValue::U32(8),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.head_count",
        MetaValue::U32(2),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.head_count_kv",
        MetaValue::Array(vec![MetaValue::U32(1), MetaValue::U32(1)]),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.sliding_window_pattern",
        MetaValue::Array(vec![MetaValue::Bool(true), MetaValue::Bool(false)]),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.key_length",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.key_length_swa",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.value_length",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.value_length_swa",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.rope.dimension_count",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.rope.dimension_count_swa",
        MetaValue::U32(4),
    );
    m(
        &mut w,
        "gemma-embedding2.rope.freq_base",
        MetaValue::F32(1e4),
    );
    m(
        &mut w,
        "gemma-embedding2.rope.freq_base_swa",
        MetaValue::F32(1e4),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-6),
    );
    m(
        &mut w,
        "gemma-embedding2.attention.sliding_window",
        MetaValue::U32(8),
    );
    m(
        &mut w,
        "gemma-embedding2.embedding_length_per_layer_input",
        MetaValue::U32(8),
    );
    m(
        &mut w,
        "gemma-embedding2.embedding_length_out",
        MetaValue::U32(8),
    );
    m(&mut w, "gemma-embedding2.pooling_type", MetaValue::U32(1));
    m(
        &mut w,
        "gemma-embedding2.attention.causal",
        MetaValue::Bool(false),
    );
    // The vocabulary block every GGUF of this family carries. No `bos`/`eos`
    // here: the `gemma4` rule set requires both and the GPT-2 bodies state
    // neither, so the tests below state them where they belong to the case.
    m(
        &mut w,
        "tokenizer.ggml.model",
        MetaValue::String("gemma4".into()),
    );
    m(
        &mut w,
        "tokenizer.ggml.tokens",
        MetaValue::Array(
            ["a", "b", "c"]
                .map(|t| MetaValue::String(t.into()))
                .to_vec(),
        ),
    );
    m(
        &mut w,
        "tokenizer.ggml.token_type",
        MetaValue::Array(vec![MetaValue::U32(1); 3]),
    );
    m(
        &mut w,
        "tokenizer.ggml.merges",
        MetaValue::Array(vec![MetaValue::String("a b".into())]),
    );
    for (key, value) in overrides {
        w.set_meta(key.to_string(), value.clone());
    }
    let f32_tensor = |w: &mut GgufWriter, name: &str, dims: [usize; 2]| {
        let bytes = vec![0u8; dims[0] * dims[1] * 4];
        w.add_tensor_bytes(name, dims.to_vec(), GgmlType::F32, bytes)
            .expect("a dense tensor");
    };
    let f32_vector = |w: &mut GgufWriter, name: &str, n: usize| {
        let bytes = vec![0u8; n * 4];
        w.add_tensor_bytes(name, vec![n], GgmlType::F32, bytes)
            .expect("a dense vector");
    };
    f32_tensor(&mut w, "token_embd.weight", [8, 3]);
    f32_tensor(&mut w, "output.weight", [8, 8]);
    f32_vector(&mut w, "output_norm.weight", 8);
    f32_tensor(&mut w, "per_layer_model_proj.weight", [8, 16]);
    f32_vector(&mut w, "per_layer_proj_norm.weight", 8);
    for il in 0..2 {
        let p = format!("blk.{il}");
        f32_tensor(&mut w, &format!("{p}.attn_q.weight"), [8, 8]);
        f32_tensor(&mut w, &format!("{p}.attn_k.weight"), [8, 4]);
        f32_tensor(&mut w, &format!("{p}.attn_v.weight"), [8, 4]);
        f32_tensor(&mut w, &format!("{p}.attn_output.weight"), [8, 8]);
        f32_vector(&mut w, &format!("{p}.attn_q_norm.weight"), 4);
        f32_vector(&mut w, &format!("{p}.attn_k_norm.weight"), 4);
        f32_vector(&mut w, &format!("{p}.attn_norm.weight"), 8);
        f32_vector(&mut w, &format!("{p}.post_attention_norm.weight"), 8);
        f32_vector(&mut w, &format!("{p}.ffn_norm.weight"), 8);
        f32_tensor(&mut w, &format!("{p}.ffn_gate.weight"), [8, 8]);
        f32_tensor(&mut w, &format!("{p}.ffn_up.weight"), [8, 8]);
        f32_tensor(&mut w, &format!("{p}.ffn_down.weight"), [8, 8]);
        f32_vector(&mut w, &format!("{p}.post_ffw_norm.weight"), 8);
        f32_tensor(&mut w, &format!("{p}.inp_gate.weight"), [8, 8]);
        f32_tensor(&mut w, &format!("{p}.proj.weight"), [8, 8]);
        f32_vector(&mut w, &format!("{p}.post_norm.weight"), 8);
        f32_vector(&mut w, &format!("{p}.layer_output_scale.weight"), 1);
    }
    mutate(&mut w);
    w
}

/// Write the synthetic file to a temp path, open it, and remove the file — the
/// loader takes a path because the real files are mmapped, so the test writes
/// one and cleans up after itself. Only whether it loaded and why it did not
/// are observed: a `Model` is not printable and nothing here reads one.
fn open(w: &GgufWriter) -> Result<Option<Model>, String> {
    let path = std::env::temp_dir().join(format!(
        "mahbot-gemma-{}-{:p}.gguf",
        std::process::id(),
        w as *const _
    ));
    let mut buf = Cursor::new(Vec::new());
    w.write(&mut buf).expect("the synthetic file writes");
    std::fs::write(&path, buf.into_inner()).expect("the synthetic file lands");
    let out = Model::open(&path).map(Some).map_err(|e| format!("{e:#}"));
    let _ = std::fs::remove_file(&path);
    out
}

/// The refusal message a synthetic file produces, or a panic naming the file as
/// one that loaded when it should have been refused.
fn refusal(w: &GgufWriter) -> String {
    match open(w) {
        Ok(_) => panic!("the synthetic file loaded when it should have been refused"),
        Err(message) => message,
    }
}

/// The synthetic file itself is a valid file: without this the refusals below
/// could be failing for an unrelated reason.
#[test]
fn the_synthetic_gemma_file_loads() {
    let model = open(&synthetic(&[], &|_| {}))
        .expect("the baseline file loads")
        .expect("the file is there");
    let g = model.gemma();
    assert_eq!((g.layers, g.d, g.heads), (2, 8, 2));
    assert_eq!(g.head_dim_of(0), 4);
    assert!(g.is_sliding(0));
    assert!(!g.is_sliding(1));
    assert_eq!(g.window_radius(), 4);
    // The file is the embeddings role, and it carries neither a body config nor
    // a `clip` one: those describe files this is not.
    assert_eq!(model.role, Some(mahbot_inference::loader::Role::Embeddings));
    assert!(model.config.is_none() && model.clip.is_none());
}

#[test]
fn a_per_layer_array_of_the_wrong_length_is_refused() {
    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.head_count_kv",
            MetaValue::Array(vec![MetaValue::U32(1)]),
        )],
        &|_| {},
    ));
    assert!(err.contains("has 1 entries"), "{err}");

    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.sliding_window_pattern",
            MetaValue::Array(vec![MetaValue::Bool(true)]),
        )],
        &|_| {},
    ));
    assert!(err.contains("sliding_window_pattern"), "{err}");
}

#[test]
fn a_value_width_that_is_not_the_key_width_is_refused() {
    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.value_length_swa",
            MetaValue::U32(8),
        )],
        &|_| {},
    ));
    assert!(err.contains("value_length_swa"), "{err}");
}

#[test]
fn a_pooling_other_than_mean_is_refused() {
    let err = refusal(&synthetic(
        &[("gemma-embedding2.pooling_type", MetaValue::U32(3))],
        &|_| {},
    ));
    assert!(err.contains("MEAN"), "{err}");
}

/// The metadata can be consistent with itself and still disagree with the
/// tensors: that is the case this loader refuses rather than reshaping.
#[test]
fn a_projection_that_disagrees_with_its_layer_arrays_is_refused() {
    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.head_count_kv",
            MetaValue::Array(vec![MetaValue::U32(2), MetaValue::U32(1)]),
        )],
        &|_| {},
    ));
    assert!(err.contains("blk.0.attn_k.weight"), "{err}");
}

/// A key that is present but holds the wrong type is refused by name: reading
/// it as zero would silently build a graph at width zero, and a mistyped
/// epsilon would make every RMSNorm divide by `√mean` with no floor.
#[test]
fn a_metadata_key_of_the_wrong_type_is_refused() {
    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.block_count",
            MetaValue::String("24".into()),
        )],
        &|_| {},
    ));
    assert!(
        err.contains("gemma-embedding2.block_count") && err.contains("string"),
        "{err}"
    );

    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.layer_norm_rms_epsilon",
            MetaValue::U32(1),
        )],
        &|_| {},
    ));
    assert!(
        err.contains("layer_norm_rms_epsilon") && err.contains("u32"),
        "{err}"
    );

    let err = refusal(&synthetic(
        &[("gemma-embedding2.attention.causal", MetaValue::U32(0))],
        &|_| {},
    ));
    assert!(
        err.contains("attention.causal") && err.contains("u32"),
        "{err}"
    );

    let err = refusal(&synthetic(
        &[(
            "gemma-embedding2.attention.head_count_kv",
            MetaValue::U32(1),
        )],
        &|_| {},
    ));
    assert!(
        err.contains("head_count_kv") && err.contains("u32"),
        "{err}"
    );
}

/// The epsilon is a divisor's floor: zero, a negative value and a non-finite
/// one are all refused by name rather than served as a NaN-producing norm.
#[test]
fn an_epsilon_that_is_not_positive_and_finite_is_refused() {
    for eps in [0.0, -1e-6, f32::NAN, f32::INFINITY] {
        let err = refusal(&synthetic(
            &[(
                "gemma-embedding2.attention.layer_norm_rms_epsilon",
                MetaValue::F32(eps),
            )],
            &|_| {},
        ));
        assert!(
            err.contains("layer_norm_rms_epsilon") && err.contains("epsilon"),
            "{err}"
        );
    }
}

/// The FFN width is a config value the graph multiplies at; a file whose
/// `feed_forward_length` disagrees with its own gate/up/down tensors is refused
/// rather than reshaped.
#[test]
fn a_feed_forward_width_that_disagrees_with_its_tensors_is_refused() {
    let err = refusal(&synthetic(
        &[("gemma-embedding2.feed_forward_length", MetaValue::U32(16))],
        &|_| {},
    ));
    assert!(err.contains("blk.0.ffn_gate.weight"), "{err}");

    // The hidden width is checked against the table it indexes as well as the
    // projections that consume it. The duplicate write replaces the table in
    // the parsed map; the metadata still says 8.
    let err = refusal(&synthetic(&[], &|w| {
        w.add_tensor_bytes(
            "token_embd.weight",
            vec![16, 3],
            GgmlType::F32,
            vec![0u8; 16 * 3 * 4],
        )
        .expect("a dense table");
    }));
    assert!(err.contains("token_embd.weight"), "{err}");
}

// ── the tokenizer's metadata gate on the same file ─────────────────

/// The tokenizer a synthetic file opens, or the reason it was refused.
fn tokenizer(w: &GgufWriter) -> Result<Tokenizer, String> {
    let model = open(w)?.expect("the file is there");
    Tokenizer::from_model(&model).map_err(|e| format!("{e:#}"))
}

/// The `gemma4` rule set wraps every sequence in both specials, so a file that
/// omits one is refused rather than tokenised with an id it does not state.
#[test]
fn a_gemma4_file_without_its_specials_is_refused() {
    let err = tokenizer(&synthetic(&[], &|_| {}))
        .err()
        .expect("no bos is stated");
    assert!(err.contains("tokenizer.ggml.bos_token_id"), "{err}");
}

/// Both specials stated, the file opens with them.
#[test]
fn the_gemma4_rule_set_reads_the_specials_it_states() {
    let t = tokenizer(&synthetic(
        &[
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(2)),
            ("tokenizer.ggml.eos_token_id", MetaValue::U32(1)),
        ],
        &|_| {},
    ))
    .expect("the file states both specials");
    assert_eq!((t.bos(), t.eos()), (2, 1));
}

/// A present-but-mistyped id is refused rather than read as 0.
#[test]
fn a_tokenizer_special_that_is_not_an_id_is_refused() {
    let err = tokenizer(&synthetic(
        &[("tokenizer.ggml.bos_token_id", MetaValue::F32(2.0))],
        &|_| {},
    ))
    .err()
    .expect("a float is not an id");
    assert!(err.contains("not an unsigned integer"), "{err}");
}

/// A rule set named with the wrong value type is refused the same way a missing
/// one is: the gate reads a string, and an integer is not a name it can match.
#[test]
fn a_tokenizer_rule_set_that_is_not_a_string_is_refused() {
    let err = tokenizer(&synthetic(
        &[("tokenizer.ggml.model", MetaValue::U32(1))],
        &|_| {},
    ))
    .err()
    .expect("an integer is not a rule set name");
    assert!(
        err.contains("`tokenizer.ggml.model` is missing or not a string"),
        "{err}"
    );
}

/// A rule set this crate does not implement is refused by name: reading the
/// vocabulary through another pre-tokenizer would tokenise every sequence into
/// different ids under the right model hash.
#[test]
fn an_unimplemented_tokenizer_rule_set_is_refused() {
    let err = tokenizer(&synthetic(
        &[("tokenizer.ggml.model", MetaValue::String("llama".into()))],
        &|_| {},
    ))
    .err()
    .expect("not a rule set this crate reads");
    assert!(
        err.contains("llama") && err.contains("does not implement"),
        "{err}"
    );
}

/// The GPT-2 bodies state no `bos` — the TTS file states an `eos` and nothing
/// else — so an omitted id is 0 there and the file opens, which is the shape the
/// `speak` cell's own file has.
#[test]
fn the_gpt2_rule_set_needs_no_bos() {
    let t = tokenizer(&synthetic(
        &[
            ("tokenizer.ggml.model", MetaValue::String("gpt2".into())),
            ("tokenizer.ggml.eos_token_id", MetaValue::U32(154_086)),
        ],
        &|_| {},
    ))
    .expect("the gpt2 rule set reads no bos");
    assert_eq!((t.bos(), t.eos()), (0, 154_086));
}
