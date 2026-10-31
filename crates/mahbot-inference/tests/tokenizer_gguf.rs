// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tokenizer's metadata gate, on a synthetic file of a surviving
//! architecture: a rule set this crate does not implement, a name or an id of the
//! wrong type, and the GPT-2 bodies that state no BOS are each pinned here, with
//! no model file and no device.

use std::io::Cursor;

use mahbot_inference::loader::Model;
use mahbot_inference::tokenizer::Tokenizer;
use rlx_gguf::{GgmlType, GgufWriter, MetaValue};

/// The smallest `qwen3tts` body the loader accepts: one layer of width 8 and a
/// three-token vocabulary whose rule set the tests below override.
fn synthetic(overrides: &[(&str, MetaValue)]) -> GgufWriter {
    let mut w = GgufWriter::new();
    w.set_arch("qwen3tts");
    let base = [
        ("qwen3tts.block_count", MetaValue::U32(1)),
        ("qwen3tts.embedding_length", MetaValue::U32(8)),
        ("qwen3tts.feed_forward_length", MetaValue::U32(8)),
        ("qwen3tts.attention.head_count", MetaValue::U32(1)),
        ("qwen3tts.attention.head_count_kv", MetaValue::U32(1)),
        (
            "qwen3tts.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-6),
        ),
        ("qwen3tts.rope.freq_base", MetaValue::F32(1e4)),
        ("tokenizer.ggml.model", MetaValue::String("gpt2".into())),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                ["a", "b", "c"]
                    .map(|t| MetaValue::String(t.into()))
                    .to_vec(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            MetaValue::Array(vec![MetaValue::U32(1); 3]),
        ),
        (
            "tokenizer.ggml.merges",
            MetaValue::Array(vec![MetaValue::String("a b".into())]),
        ),
    ];
    // The defaults first, then the test's own keys over them.
    for (key, value) in base
        .into_iter()
        .chain(overrides.iter().map(|(key, value)| (*key, value.clone())))
    {
        w.set_meta(key.to_string(), value);
    }
    let mut dense = |name: &str, dims: &[usize]| {
        let bytes = vec![0u8; dims.iter().product::<usize>() * 4];
        w.add_tensor_bytes(name, dims.to_vec(), GgmlType::F32, bytes)
            .expect("a dense tensor");
    };
    dense("token_embd.weight", &[8, 3]);
    dense("output.weight", &[8, 3]);
    dense("output_norm.weight", &[8]);
    w
}

/// The tokenizer a synthetic file opens, or the reason it was refused.
fn tokenizer(w: &GgufWriter) -> Result<Tokenizer, String> {
    let path = std::env::temp_dir().join(format!(
        "mahbot-tokenizer-{}-{:p}.gguf",
        std::process::id(),
        w as *const _
    ));
    let mut buf = Cursor::new(Vec::new());
    w.write(&mut buf).expect("the synthetic file writes");
    std::fs::write(&path, buf.into_inner()).expect("the synthetic file lands");
    let out = Model::open(&path)
        .and_then(|model| Tokenizer::from_model(&model))
        .map_err(|e| format!("{e:#}"));
    let _ = std::fs::remove_file(&path);
    out
}

/// The synthetic file itself loads: without this the refusals below could be
/// failing for an unrelated reason.
#[test]
fn the_synthetic_file_loads() {
    tokenizer(&synthetic(&[])).expect("the baseline file opens");
}

/// A present-but-mistyped id is refused rather than read as 0.
#[test]
fn a_tokenizer_special_that_is_not_an_id_is_refused() {
    let err = tokenizer(&synthetic(&[(
        "tokenizer.ggml.bos_token_id",
        MetaValue::F32(2.0),
    )]))
    .err()
    .expect("a float is not an id");
    assert!(err.contains("not an unsigned integer"), "{err}");
}

/// A rule set named with the wrong value type is refused the same way a missing
/// one is: the gate reads a string, and an integer is not a name it can match.
#[test]
fn a_tokenizer_rule_set_that_is_not_a_string_is_refused() {
    let err = tokenizer(&synthetic(&[("tokenizer.ggml.model", MetaValue::U32(1))]))
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
    let err = tokenizer(&synthetic(&[(
        "tokenizer.ggml.model",
        MetaValue::String("llama".into()),
    )]))
    .err()
    .expect("not a rule set this crate reads");
    assert!(
        err.contains("llama") && err.contains("does not implement"),
        "{err}"
    );
}

/// The GPT-2 bodies state no `bos` — the TTS file states an `eos` and nothing
/// else — so an omitted id is 0 there and the file opens.
#[test]
fn the_gpt2_rule_set_needs_no_bos() {
    let t = tokenizer(&synthetic(&[(
        "tokenizer.ggml.eos_token_id",
        MetaValue::U32(154_086),
    )]))
    .expect("the gpt2 rule set reads no bos");
    assert_eq!((t.bos(), t.eos()), (0, 154_086));
}
