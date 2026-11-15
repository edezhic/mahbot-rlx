// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tokenizer that lives in the GGUF itself.
//!
//! The Qwen files carry `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` — the
//! same GPT-2 byte-level BPE the reference uses. [`Encoder`] reads those two
//! tables and applies them: pre-tokenize, map UTF-8 bytes to the byte-level
//! alphabet, merge the lowest-ranked adjacent pair until the word is stable, and
//! look the final pieces up in the vocabulary. [`Decoder`] is the reverse, with
//! the control tokens left out of the text.
//!
//! **One half per path.** Each direction reads its own tables, and each half
//! builds only those: synthesis encodes and never decodes, recognition decodes
//! and never encodes — its prompt ids are fixed constants. Measured on the two
//! 1.7B bodies, opening one and building a whole vocabulary costs 70.9 MiB of
//! physical footprint on recognition and 71.3 MiB on synthesis; with one half
//! built, recognition retains 1.8 MiB and synthesis 47.2 MiB.
//!
//! The tables themselves are read out of the file's mapping, not from the
//! metadata [`Model::open`] parsed: the open leaves those three arrays in the
//! file (see [`rlx_gguf::GgufFile::skipped`]) rather than materialising them, so
//! the ear path — which builds no tokenizer at all — holds none of it.
//!
//! The file names its own rule set in `tokenizer.ggml.model`, and a name this
//! module does not implement is refused rather than read through the byte
//! alphabet: tokenising a sequence with another pre-tokenizer produces different
//! ids under the right model hash, which no later check would catch. `gpt2` is
//! the one name this crate reads.
//!
//! [`Encoder::encode`] never adds the special tokens: a caller states its own
//! convention around the returned ids.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{MetaValue, Skipped};

use crate::loader::Model;

/// Text → ids: the half the synthesis path reads.
pub struct Encoder {
    ids: HashMap<String, usize>,
    /// `(left, right) -> rank`; a lower rank is merged first.
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
}

/// Ids → text: the half the recognition path reads.
pub struct Decoder {
    /// Every token's bytes, mapped out of the byte alphabet at build time and
    /// concatenated in id order.
    text: Vec<u8>,
    /// `starts[i]..starts[i + 1]` is token `i`'s slice of `text`.
    starts: Vec<u32>,
    /// The file's `tokenizer.ggml.token_type`, as "is this an ordinary token"
    /// (`1`): unknown, control, user-defined, unused and byte tokens are not text.
    ordinary: Vec<bool>,
}

impl Encoder {
    pub fn from_model(model: &Model) -> Result<Self> {
        // The rule set is checked before a table is read: a vocabulary this
        // crate cannot apply is refused without building one.
        rule_set(model)?;
        check_specials(model)?;
        let tokens = table(model, "tokenizer.ggml.tokens")?;
        let mut ids = HashMap::with_capacity(tokens.count());
        for (id, token) in tokens.strings()?.enumerate() {
            ids.insert(token?.to_string(), id);
        }
        let pairs = table(model, "tokenizer.ggml.merges")?;
        let mut merges = HashMap::with_capacity(pairs.count());
        for (rank, item) in pairs.strings()?.enumerate() {
            let pair = item?;
            // llama.cpp finds the separator from the second byte on
            // (`strchr(word + 1, ' ')`), so a one-character first token —
            // including `" "` itself — cannot be mistaken for an empty one.
            let at = pair
                .char_indices()
                .find(|(i, c)| *c == ' ' && *i >= 1)
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow!("merge `{pair}` has no separator"))?;
            // First writer wins, as `std::unordered_map::emplace` does: this
            // vocabulary's merge list is not guaranteed free of a repeated pair,
            // and taking the later rank would reorder merges it shares.
            merges
                .entry((pair[..at].to_string(), pair[at + 1..].to_string()))
                .or_insert(rank as u32);
        }
        Ok(Self {
            ids,
            merges,
            byte_to_char: byte_to_char(),
        })
    }

    /// Encode text into the model's ids, **without** BOS/EOS. The pre-tokenizer
    /// is the Qwen2/GPT-2 one (a word with its leading space, a single digit, a
    /// punctuation run, or a whitespace run).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for piece in pretokenize(text) {
            out.extend(self.encode_piece(&piece));
        }
        out
    }

    fn encode_piece(&self, piece: &str) -> Vec<usize> {
        let mut word: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect();
        while word.len() > 1 {
            let mut best: Option<(u32, (String, String))> = None;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if best.as_ref().is_none_or(|(best_rank, _)| rank < *best_rank) {
                        best = Some((rank, pair));
                    }
                }
            }
            let Some((_, (a, b))) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word.into_iter()
            .filter_map(|w| self.ids.get(&w).copied())
            .collect()
    }
}

impl Decoder {
    pub fn from_model(model: &Model) -> Result<Self> {
        rule_set(model)?;
        check_specials(model)?;
        let tokens = table(model, "tokenizer.ggml.tokens")?;
        let kinds = table(model, "tokenizer.ggml.token_type")?;
        let back = char_to_byte();
        // Every token's bytes and nothing else: an upper bound on the decoded
        // text, reserved once and never regrown.
        let mut text = Vec::with_capacity(tokens.text_len());
        let mut ordinary = Vec::with_capacity(kinds.count());
        let mut starts = Vec::with_capacity(tokens.count() + 1);
        starts.push(0u32);
        for token in tokens.strings()? {
            for ch in token?.chars() {
                if let Some(&byte) = back.get(&ch) {
                    text.push(byte);
                }
            }
            starts.push(u32::try_from(text.len()).context("the vocabulary is wider than 4 GiB")?);
        }
        for kind in kinds.u32s()? {
            ordinary.push(kind? == 1);
        }
        Ok(Self {
            text,
            starts,
            ordinary,
        })
    }

    /// The generated text. Control tokens (the chat/audio markers) are skipped;
    /// every ordinary token goes back through the byte alphabet, which
    /// [`Self::from_model`] already undid.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let (Some(&from), Some(&to)) =
                (self.starts.get(id), self.starts.get(id.saturating_add(1)))
            else {
                continue;
            };
            if !self.ordinary.get(id).copied().unwrap_or(true) {
                continue;
            }
            bytes.extend_from_slice(&self.text[from as usize..to as usize]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// A vocabulary table, out of the file's mapping rather than the open's metadata
/// — see the module doc. Present only when the open deferred that key
/// ([`Model::open`] defers all three); an open that materialised it, or a file
/// that writes it as something other than an array, has nothing to serve here and
/// is refused rather than materialised on demand.
fn table<'m>(model: &'m Model, key: &str) -> Result<Skipped<'m>> {
    model
        .file
        .skipped(key)
        .ok_or_else(|| anyhow!("`{key}` is missing or not an array"))
}

/// The file names its own rule set, and another one is not interchangeable: an
/// unknown name is a vocabulary this crate cannot read, not a reason to read it
/// through the byte alphabet.
fn rule_set(model: &Model) -> Result<()> {
    let kind = model
        .file
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(MetaValue::as_str)
        .ok_or_else(|| anyhow!("`tokenizer.ggml.model` is missing or not a string"))?;
    if kind != "gpt2" {
        bail!("`tokenizer.ggml.model` is `{kind}`, a rule set this crate does not implement");
    }
    Ok(())
}

/// A present-but-mistyped prompt id is a malformed file and is refused here. The
/// ids themselves are not kept: neither half reads them — the talker prompt and
/// the recognition markers state their own.
fn check_specials(model: &Model) -> Result<()> {
    for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        if let Some(value) = model.file.metadata.get(key) {
            value
                .as_u64()
                .ok_or_else(|| anyhow!("`{key}` is present but not an unsigned integer"))?;
        }
    }
    Ok(())
}

/// The GPT-2 byte-level alphabet: every byte, in the reference's order, paired
/// with the char that stands for it. `b'!'..=b'~'`, `0xA1..=0xAC` and
/// `0xAE..=0xFF` stand for themselves; the rest are `U+0100 + n`.
fn alphabet() -> [(u8, char); 256] {
    let mut pairs: Vec<(u8, char)> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .map(|b| (b, b as char))
        .collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !pairs.iter().any(|(known, _)| *known == b) {
            pairs.push((b, char::from_u32(256 + n).expect("a scalar value")));
            n += 1;
        }
    }
    pairs.try_into().expect("the alphabet is 256 bytes")
}

fn byte_to_char() -> [char; 256] {
    let mut map = [char::REPLACEMENT_CHARACTER; 256];
    for (b, c) in alphabet() {
        map[b as usize] = c;
    }
    map
}

fn char_to_byte() -> HashMap<char, u8> {
    alphabet().into_iter().map(|(b, c)| (c, b)).collect()
}

fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' && i + 1 < chars.len() && chars[i + 1].is_alphabetic() {
            let start = i;
            i += 2;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c == ' '
            && i + 1 < chars.len()
            && !chars[i + 1].is_whitespace()
            && !chars[i + 1].is_alphabetic()
            && !chars[i + 1].is_numeric()
        {
            let start = i;
            i += 1;
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c == '\'' && i + 1 < chars.len() {
            // GPT-2/Qwen2's contraction alternative (`'s`, `'t`, `'re`, …) is
            // one pre-token, not punctuation followed by a word: `people's` is
            // ` people` + `'s`, and the merge table has `'`+`s`.
            let rest: String = chars[i + 1..].iter().take(3).collect();
            let lower = rest.to_ascii_lowercase();
            let take =
                if lower.starts_with("re") || lower.starts_with("ve") || lower.starts_with("ll") {
                    2
                } else if lower.starts_with('s')
                    || lower.starts_with('t')
                    || lower.starts_with('m')
                    || lower.starts_with('d')
                {
                    1
                } else {
                    0
                };
            if take > 0 {
                out.push(chars[i..i + 1 + take].iter().collect());
                i += 1 + take;
            } else {
                let start = i;
                i += 1;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !chars[i].is_alphabetic()
                    && !chars[i].is_numeric()
                {
                    i += 1;
                }
                out.push(chars[start..i].iter().collect());
            }
        } else if c.is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c.is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c.is_numeric() {
            out.push(c.to_string());
            i += 1;
        } else {
            let start = i;
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs the synthesis pair installed; without it the test prints that it did
    /// not run and returns, so it proves nothing here.
    #[test]
    fn qwen3_tts_known_phrase() {
        // The models are inputs, not repository content: a machine without them
        // skips this check instead of failing on a path the checkout lacks.
        let Ok(dir) = crate::loader::models_dir() else {
            eprintln!("skipped: the model directory is not installed");
            return;
        };
        let Some((body, _)) = crate::loader::pair_files(crate::loader::Role::Synthesis) else {
            return;
        };
        let path = dir.join(body);
        if !path.exists() {
            eprintln!("skipped: {} is not on this machine", path.display());
            return;
        }
        let model = Model::open(path).unwrap();
        let t = Encoder::from_model(&model).unwrap();
        assert_eq!(
            t.encode("Hello world. This is a test."),
            vec![9707, 1879, 13, 1096, 374, 264, 1273, 13]
        );
        assert_eq!(t.encode("people's"), vec![16069, 594]);
    }
}
