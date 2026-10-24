// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `gemma4` vocabulary's rules on a hand-built vocabulary: the space
//! escaping, the newline-only pre-tokenizer, the whole-newline-run lookup and
//! the `<0xXX>` byte fallback. No model file and no device — this target is the
//! one that can run anywhere, so it is where the rules the embedding cells
//! depend on are pinned.
//!
//! What it does not establish: that the rules match llama.cpp on real text. That
//! is `tests/embed_model.rs`, which needs the GGUF and compares the article's
//! ids against the reference engine's, token for token.
//!
//! The rule set is ported from `llama-vocab.cpp`'s `LLAMA_VOCAB_PRE_TYPE_GEMMA4`
//! (`escape_whitespaces = true`, `byte_encode = false`, `"[^\n]+|[\n]+"`, the
//! whole-run newline lookup of ggml-org/llama.cpp#21343). Note that the
//! metaspace block that prefixes a word with `▁` and splits again on it is
//! guarded by `PRE_TYPE_MMBERT` in that file and does **not** apply here: for
//! gemma4 the escape happens once, over the whole text, before the split.

use mahbot_inference::tokenizer::Tokenizer;

/// `<pad>`, `<eos>`, `<bos>`, the characters the rules need, then the three byte
/// tokens that spell U+2581.
fn tiny() -> Tokenizer {
    Tokenizer::for_test(
        &[
            "<pad>", "<eos>", "<bos>", "▁", "a", "b", "ab", "▁a", "▁ab", "\n", "\n\n", "<0xE2>",
            "<0x96>", "<0x81>",
        ],
        &[("a", "b"), ("▁", "a"), ("▁a", "b"), ("\n", "\n")],
        2,
        1,
    )
}

/// The space is escaped **before** the text is split, so `"a a"` is `a` then
/// `▁a`: the second word's leading space is part of the word, and a merge that
/// begins with `▁` reaches across it.
#[test]
fn a_space_becomes_a_bar_inside_the_word() {
    let t = tiny();
    assert_eq!(t.encode("a a"), vec![4, 7]);
    assert_eq!(t.encode("ab a"), vec![6, 7]);
    assert_eq!(t.encode(" a"), vec![7]);
    // The **lowest** rank merges first: `("a","b")` is rank 0 and `("▁","a")`
    // rank 1 in this table, so `▁ab` becomes `▁` + `ab`, not `▁a` + `b`.
    assert_eq!(t.encode("a ab"), vec![4, 3, 6]);
    // Rank the other way round and the same text merges the other way: the rule
    // is the table's order, not the character classes.
    let flipped = Tokenizer::for_test(
        &["<pad>", "<eos>", "<bos>", "▁", "a", "b", "ab", "▁a", "▁ab"],
        &[("▁", "a"), ("▁a", "b"), ("a", "b")],
        2,
        1,
    );
    assert_eq!(flipped.encode("a ab"), vec![4, 8]);
}

/// No merge spans a newline, and a run of nothing but newlines is looked up as
/// a whole first: `"\n\n"` is the token the vocabulary holds for it, not the
/// `"\n"`+`"\n"` merge, which is what makes the lookup observable at all.
#[test]
fn a_newline_run_is_looked_up_whole() {
    let t = tiny();
    assert_eq!(t.encode("\n"), vec![9]);
    assert_eq!(t.encode("\n\n"), vec![10]);
    assert_eq!(t.encode("a\n\na"), vec![4, 10, 4]);
    // Three newlines are not in the vocabulary: the merge takes them to
    // `"\n\n"` + `"\n"`.
    assert_eq!(t.encode("\n\n\n"), vec![10, 9]);
}

/// There is no byte alphabet: a symbol the vocabulary does not hold falls back
/// to one `<0xXX>` token per UTF-8 byte, in uppercase hex.
#[test]
fn an_unknown_symbol_falls_back_to_its_bytes() {
    // A vocabulary that holds `▁` matches it directly...
    assert_eq!(tiny().encode("▁"), vec![3]);
    // ...and one that does not answers U+2581's three bytes.
    let bare = Tokenizer::for_test(
        &["<pad>", "<eos>", "<bos>", "a", "<0xE2>", "<0x96>", "<0x81>"],
        &[],
        2,
        1,
    );
    assert_eq!(bare.encode("▁"), vec![4, 5, 6]);
}

/// `encode` never adds the special tokens; `encode_with_specials` puts the
/// vocabulary's own BOS and EOS around them, once each.
#[test]
fn the_special_tokens_are_the_callers_choice() {
    let t = tiny();
    assert_eq!(t.encode("ab"), vec![6]);
    assert_eq!(t.encode_with_specials("ab"), vec![2, 6, 1]);
    assert_eq!(t.bos(), 2);
    assert_eq!(t.eos(), 1);
}
