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
//!
//! The embeddings file names a different vocabulary (`tokenizer.ggml.model =
//! "gemma4"`) and needs a different rule set, ported from llama.cpp's
//! `llama-vocab.cpp` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`):
//!
//! * **the space is escaped first, over the whole text** — every `' '` becomes
//!   U+2581 `▁` (`escape_whitespaces = true`), which is why a word's leading
//!   space is part of the word and why merges cross word boundaries;
//! * **the pre-tokenizer is `[^\n]+|[\n]+`** — the text splits into runs of
//!   non-newlines and runs of newlines, and no merge spans a newline;
//! * **a run of nothing but newlines is looked up whole** first (llama.cpp's
//!   fix for gemma 4, ggml-org/llama.cpp#21343): `"\n\n"` is one token if the
//!   vocabulary has one, and only otherwise split into merge-able characters;
//! * **no byte alphabet** (`byte_encode = false`): symbols are UTF-8 characters
//!   and the merges are over raw text, so an unknown piece falls back to
//!   `<0xXX>` byte tokens with uppercase hex, one per byte;
//! * **BOS is forced** for this model whatever `add_bos_token` says
//!   (`llama-vocab.cpp` around line 2970), and this file also asks for EOS.
//!
//! [`Tokenizer::encode`] never adds the special tokens: a cell states its own
//! convention through [`Tokenizer::encode_with_specials`], which is what the
//! embedding cells use.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use anyhow::{Result, anyhow};
use rlx_gguf::MetaValue;

use crate::loader::Model;

/// Which vocabulary a [`Tokenizer`] speaks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// GPT-2 byte-level BPE — the Qwen bodies.
    Gpt2,
    /// llama.cpp's `gemma4` — raw UTF-8 BPE over `▁`-escaped text.
    Gemma4,
}

pub struct Tokenizer {
    kind: Kind,
    tokens: Vec<String>,
    token_type: Vec<u32>,
    ids: HashMap<String, usize>,
    /// `(left, right) -> rank`; a lower rank is merged first.
    merges: HashMap<(String, String), u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    bos: usize,
    eos: usize,
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
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        let (byte_to_char, char_to_byte) = byte_alphabet();
        // The file names its own rule set, and the two are not interchangeable:
        // an unknown name is a vocabulary this crate cannot read, not a reason to
        // read it through the byte alphabet.
        let model_kind = model
            .file
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| anyhow!("`tokenizer.ggml.model` is missing or not a string"))?;
        let kind = match model_kind {
            "gemma4" => Kind::Gemma4,
            "gpt2" => Kind::Gpt2,
            other => {
                return Err(anyhow!(
                    "`tokenizer.ggml.model` is `{other}`, a rule set this crate does not implement"
                ));
            }
        };
        // A present-but-mistyped id is refused rather than read as 0. The gemma4
        // path wraps every sequence in both (the embedding cells), so a file that
        // omits either cannot be tokenised the way it defines; the GPT-2 bodies
        // state their own prompt specials, and the TTS body omits `bos`.
        let special = |key: &str| -> Result<Option<usize>> {
            match model.file.metadata.get(key) {
                None => Ok(None),
                Some(value) => value
                    .as_u64()
                    .map(|v| Some(v as usize))
                    .ok_or_else(|| anyhow!("`{key}` is present but not an unsigned integer")),
            }
        };
        let bos = special("tokenizer.ggml.bos_token_id")?;
        let eos = special("tokenizer.ggml.eos_token_id")?;
        let (bos, eos) = if matches!(kind, Kind::Gemma4) {
            (
                bos.ok_or_else(|| anyhow!("`tokenizer.ggml.bos_token_id` is missing"))?,
                eos.ok_or_else(|| anyhow!("`tokenizer.ggml.eos_token_id` is missing"))?,
            )
        } else {
            (bos.unwrap_or(0), eos.unwrap_or(0))
        };
        Ok(Self {
            kind,
            tokens,
            token_type,
            ids,
            merges,
            byte_to_char,
            char_to_byte,
            bos,
            eos,
        })
    }

    /// A tokenizer over hand-built tables — the model-free tests' constructor.
    /// It builds the `gemma4` shape (raw UTF-8 symbols, no byte alphabet),
    /// which is the rule set a small vocabulary can exercise whole; the GPT-2
    /// path's pre-tokenizer needs the Qwen files and is pinned against them.
    #[doc(hidden)]
    pub fn for_test(tokens: &[&str], merges: &[(&str, &str)], bos: usize, eos: usize) -> Self {
        let tokens: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        let token_type = vec![1u32; tokens.len()];
        let merges = merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.to_string(), b.to_string()), rank as u32))
            .collect();
        let (byte_to_char, char_to_byte) = byte_alphabet();
        Self {
            kind: Kind::Gemma4,
            tokens,
            token_type,
            ids,
            merges,
            byte_to_char,
            char_to_byte,
            bos,
            eos,
        }
    }

    /// The vocabulary's BOS id.
    pub fn bos(&self) -> usize {
        self.bos
    }
    /// The vocabulary's EOS id.
    pub fn eos(&self) -> usize {
        self.eos
    }

    /// Encode text into the model's ids, **without** BOS/EOS. The pre-tokenizer
    /// is the file's own: the Qwen2/GPT-2 one for the bodies (a word with its
    /// leading space, a single digit, a punctuation run, or a whitespace run),
    /// and llama.cpp's gemma4 rules for the embeddings file.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.kind {
            Kind::Gpt2 => {
                let mut out = Vec::new();
                for piece in pretokenize(text) {
                    out.extend(self.encode_piece(&piece));
                }
                out
            }
            Kind::Gemma4 => self.encode_gemma4(text),
        }
    }

    /// [`Tokenizer::encode`] with the model's own BOS and EOS around it.
    pub fn encode_with_specials(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(2);
        ids.push(self.bos);
        ids.extend(self.encode(text));
        ids.push(self.eos);
        ids
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

    /// llama.cpp's gemma4 pass: escape the spaces over the whole text, split on
    /// newlines only, and BPE each run over its UTF-8 characters.
    fn encode_gemma4(&self, text: &str) -> Vec<usize> {
        let escaped = text.replace(' ', "\u{2581}");
        let mut out = Vec::new();
        let mut run = String::new();
        let mut in_newlines = false;
        let flush = |run: &mut String, out: &mut Vec<usize>| {
            if !run.is_empty() {
                self.push_gemma4_run(run, out);
                run.clear();
            }
        };
        for ch in escaped.chars() {
            let newline = ch == '\n';
            if newline != in_newlines {
                flush(&mut run, &mut out);
                in_newlines = newline;
            }
            run.push(ch);
        }
        flush(&mut run, &mut out);
        out
    }

    /// One `[^\n]+` or `[\n]+` run: the whole-run lookup first (a run of only
    /// newlines that the vocabulary holds is one token), then a plain BPE.
    fn push_gemma4_run(&self, run: &str, out: &mut Vec<usize>) {
        if run.bytes().all(|b| b == b'\n')
            && let Some(&id) = self.ids.get(run)
        {
            out.push(id);
            return;
        }
        let symbols: Vec<String> = run.chars().map(|c| c.to_string()).collect();
        for piece in self.merge(symbols) {
            match self.ids.get(&piece) {
                Some(&id) => out.push(id),
                None => {
                    // No byte alphabet here: a piece the vocabulary does not
                    // hold falls back to one `<0xXX>` token per byte.
                    for b in piece.bytes() {
                        let token = format!("<0x{b:02X}>");
                        if let Some(&id) = self.ids.get(&token) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Merge the lowest-ranked adjacent pair until no pair has a rank, then
    /// return the symbols left — llama.cpp's `work_queue` walk for the gemma4
    /// vocabulary.
    ///
    /// Each heap entry carries the **concatenation it was pushed for**, not only
    /// the index: a symbol list mutates under the queue, so the pair now sitting
    /// at an entry's index can be a different pair whose own rank is higher, and
    /// merging it then would take a merge out of order. llama.cpp guards this by
    /// comparing the joined text (`llama-vocab.cpp`, the `left_token +
    /// right_token != bigram.text` test) and this does the same — including its
    /// one consequence, that a pair re-split at another point but joining to the
    /// same text merges at the entry's rank.
    ///
    /// (The GPT-2 path above keeps its own lowest-rank-then-merge-all-occurrences
    /// loop: the two orders agree on the Qwen vocabularies' inputs, but that is a
    /// measured fact about those texts and not a property to rely on, so neither
    /// path was made to serve both.)
    fn merge(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }
        let mut prev: Vec<isize> = (0..symbols.len() as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..symbols.len() as isize).map(|i| i + 1).collect();
        *next.last_mut().expect("a symbol") = -1;
        let mut heap: BinaryHeap<Reverse<(u32, usize, String)>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<(u32, usize, String)>>,
                    symbols: &Vec<String>,
                    left: isize,
                    right: isize| {
            if left < 0 || right < 0 {
                return;
            }
            let pair = (
                symbols[left as usize].clone(),
                symbols[right as usize].clone(),
            );
            if let Some(&rank) = self.merges.get(&pair) {
                let text = format!("{}{}", pair.0, pair.1);
                heap.push(Reverse((rank, left as usize, text)));
            }
        };
        for i in 1..symbols.len() {
            push(&mut heap, &symbols, i as isize - 1, i as isize);
        }
        while let Some(Reverse((_, left, text))) = heap.pop() {
            let right = next[left];
            if right < 0 {
                continue;
            }
            let right = right as usize;
            if format!("{}{}", symbols[left], symbols[right]) != text {
                continue;
            }
            symbols[left] = text;
            symbols[right].clear();
            let after = next[right];
            next[left] = after;
            if after >= 0 {
                prev[after as usize] = left as isize;
            }
            let before = prev[left];
            push(&mut heap, &symbols, before, left as isize);
            push(&mut heap, &symbols, left as isize, after);
        }
        symbols.retain(|s| !s.is_empty());
        symbols
    }

    /// The generated text. Control tokens (the chat/audio markers) are skipped;
    /// every ordinary token goes back through the byte alphabet — or, for the
    /// gemma4 vocabulary, back through its own `▁` spelling.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else {
                continue;
            };
            if self.token_type.get(id).copied().unwrap_or(1) != 1 {
                continue;
            }
            if self.kind == Kind::Gemma4 {
                let text = tok.replace('\u{2581}', " ");
                bytes.extend_from_slice(text.as_bytes());
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
