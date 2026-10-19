// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tokenizer that lives in the GGUF itself.
//!
//! The Qwen files carry `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` — the
//! same GPT-2 byte-level BPE the reference uses. This module reads those two
//! tables and applies them: pre-tokenize, map UTF-8 bytes to the byte-level
//! alphabet, merge the lowest-ranked adjacent pair until the word is stable, and
//! look the final pieces up in the vocabulary. Decoding is the reverse, with the
//! control tokens left out of the text.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rlx_gguf::MetaValue;

use crate::loader::Model;

pub struct Tokenizer {
    tokens: Vec<String>,
    token_type: Vec<u32>,
    ids: HashMap<String, usize>,
    /// `(left, right) -> rank`; a lower rank is merged first.
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn from_model(model: &Model) -> Result<Self> {
        let arrays = |key: &str| -> Result<&[MetaValue]> {
            model
                .file
                .metadata
                .get(key)
                .and_then(MetaValue::as_array)
                .ok_or_else(|| anyhow!("`{key}` is missing"))
        };
        let tokens: Vec<String> = arrays("tokenizer.ggml.tokens")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("token is not a string"))
            })
            .collect::<Result<_>>()?;
        let token_type: Vec<u32> = arrays("tokenizer.ggml.token_type")?
            .iter()
            .map(|v| v.as_u32().ok_or_else(|| anyhow!("token type is not a u32")))
            .collect::<Result<_>>()?;
        let mut merges = HashMap::new();
        for (rank, item) in arrays("tokenizer.ggml.merges")?.iter().enumerate() {
            let pair = item
                .as_str()
                .ok_or_else(|| anyhow!("merge is not a string"))?;
            let (a, b) = pair
                .split_once(' ')
                .ok_or_else(|| anyhow!("merge `{pair}` has no separator"))?;
            merges.insert((a.to_string(), b.to_string()), rank as u32);
        }
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        let (byte_to_char, char_to_byte) = byte_alphabet();
        Ok(Self {
            tokens,
            token_type,
            ids,
            merges,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Encode text into the model's ids. The pre-tokenizer is the Qwen2/GPT-2
    /// one for the cases model texts use: a word with its leading space,
    /// a single digit, a punctuation run (also with its leading space), or a
    /// whitespace run.
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

    /// The generated text. Control tokens (the chat/audio markers) are skipped;
    /// every ordinary token goes back through the byte alphabet.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            if self.token_type.get(id).copied().unwrap_or(1) != 1 {
                continue;
            }
            for ch in tok.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).expect("a scalar value"));
            n += 1;
        }
    }
    let mut map = [char::REPLACEMENT_CHARACTER; 256];
    let mut back = HashMap::new();
    for (&b, &c) in bs.iter().zip(&cs) {
        map[b as usize] = c;
        back.insert(c, b);
    }
    (map, back)
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

    /// Needs the synthesis pair in `MAHBOT_MODELS_DIR`; without it the test
    /// prints that it did not run and returns, so it proves nothing here.
    #[test]
    fn qwen3_tts_known_phrase() {
        // The models are inputs, not repository content: a machine without them
        // skips this check instead of failing on a path the checkout lacks.
        let Ok(dir) = crate::loader::models_dir() else {
            eprintln!("skipped: {} is not set", crate::loader::MODELS_DIR_ENV);
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
        let t = Tokenizer::from_model(&model).unwrap();
        assert_eq!(
            t.encode("Hello world. This is a test."),
            vec![9707, 1879, 13, 1096, 374, 264, 1273, 13]
        );
        assert_eq!(t.encode("people's"), vec![16069, 594]);
    }
}
