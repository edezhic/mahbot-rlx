// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The answers the two generative surfaces must keep producing, pinned before
//! the memory-bandwidth work started. Both are bit-exact: recognition takes an
//! argmax, synthesis samples from a seed, so a drift in the last bits flips a
//! token, then a codec frame, then the whole waveform.
//!
//! Taken from a plain run of `mahbot-inference 4` (recognition) and `5`
//! (synthesis) on the models `MEASUREMENTS.md` records, with no `RLX_*` or
//! `STAND_*` switch set: the instrumented runs are not witnesses, because a
//! switch may change the scheduling shape. Two recognition runs produced the
//! same transcript and digest; the synthesis waveform digest is the one a plain
//! run prints as the cell's own `digest=`.

use sha2::{Digest, Sha256};

/// Recognition: the clip is the recorded 17.46 s Russian reference, 1 746 mel
/// frames → a 234-row prompt (3 head + 6 text + 219 audio + 6 tail).
pub const LISTEN_FRAMES: usize = 1_746;
pub const LISTEN_PROMPT_ROWS: usize = 234;
pub const LISTEN_GENERATED: usize = 65;
// The pinned digests in this file are a **witness, not a gate** (owner-directed):
// a hash that moves because the real outputs moved by a rounding is recorded and
// reported — with what moved and by how much — and the round carries on, as long
// as the recognised transcript of the synthesised wave and the speech's
// intelligibility and range are intact. What is hard: no non-finite values and no
// clipping. Do not roll back a measured win because a digest moved in its last
// digits, and do not spend a round chasing one back.
/// `digest(generated ids ++ next ++ eos flag ++ written)` — the cell's own
/// answer digest, printed as `digest=`.
pub const LISTEN_ANSWER_DIGEST: u64 = 0x2f93_dfa1_d7cd_eeb1;
/// SHA-256 of the decoded transcript, in UTF-8.
pub const LISTEN_TRANSCRIPT_SHA256: &str =
    "522e47e1f6ed20ca69ad638df1b2b44ed03ef721e349d6ffacea4e13f7c18d3f";
pub const LISTEN_TRANSCRIPT: &str = "обычные объявления метро делались только на каталонском языке, но не забывали переводить объявления в другие языки, например, итальянский, испанский, английский, французский, итальянский";

/// Synthesis: 330 codec rows, 26.40 s at 24 kHz, `330 · 1920 = 633 600` samples.
pub const SPEAK_ROWS: usize = 330;
pub const SPEAK_SAMPLES: usize = 633_600;
/// `digest(&waveform)` — FNV-1a over the f32 bytes, the cell's own `digest=`.
///
/// Re-recorded when the sound half's `SnakeBeta` chain became one dispatch and the
/// frame's norms became 16 B loads: both move the waveform's last bits (the snake
/// by at most one count of its 638-count peak on 29 of 153 600 samples, the norms
/// by their summation order), and the previous pin was `bae7_ba02_2d33_5038`.
/// The transcript below is the gate; this digest is the record of the waveform
/// that produced it.
pub const SPEAK_WAVE_DIGEST: u64 = 0x231d_8234_9645_0627;
/// The `prompt_digest` `MEASUREMENTS.md` carried before the waveform digest was
/// pinned — the talker prompt's FNV-1a digest, recorded beside the waveform so
/// the waveform can be audited against a value that predates it.
pub const SPEAK_PROMPT_DIGEST: u64 = 0xfb16_0ff9_665b_4331;
/// The voice the speak witness was recorded with — `MEASUREMENTS.md`'s
/// provenance row. `MAHBOT_VOICE_WAV` names the voice input, so a run that
/// points it at another file expects another waveform and skips the pin.
pub const SPEAK_VOICE_SHA256: &str =
    "0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace";
/// The synthesis round trip: the pinned waveform, resampled 24 kHz → 16 kHz and
/// run back through the recogniser exactly as the listen cell runs it, comes back
/// as this text — 107 generated ids, EOS reached, on the recorded voice and the
/// models `MEASUREMENTS.md` names.
///
/// It is the **raw** waveform's transcript. The gate does not apply
/// `prepare_speech`: that is the recognition side's conditioning, and this
/// waveform's peak (0.053) trips its quiet-recording gain, which multiplies the
/// signal by ~13 before the encoder sees it. The same clip through the prepared
/// path answers differently ("…never mind how long since I left New Orleans…
/// I've been to New York…"), so the two are not interchangeable and this pin
/// belongs to the path the gate takes. What the pin establishes is that the
/// waveform still says the same thing to the recogniser; it is not a claim that
/// the synthesis is intelligible — the answer keeps the opening clause and then
/// runs a repetition loop, exactly as it does on the prepared path.
pub const SPEAK_TRANSCRIPT: &str = "Call me Ishmael. Some years ago, now I'm old, beside the sea. I've seen them all, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every ocean, every island, every shore, every harbor, every city, every town, every street, every house, every farm, every field, every hill, every mountain, every forest, every river, every";

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    crate::measurements::hex(&h.finalize())
}

// ── the embedding cells' article ───────────────────────────────────
//
// The two embedding cells read one article, pinned here rather than in the
// stand: the model file is the other input and `MEASUREMENTS.md` names both.
// The text itself is a Wikipedia extract under CC BY-SA, so the checkout
// carries its pins and not its bytes — `MAHBOT_EMBED_TEXT` names the file.

/// The revision the extract was taken from (2026-10-20T21:22:49Z).
pub const ARTICLE_REVISION: u64 = 1_381_231_733;
/// The API URL the text came from: `action=query&prop=extracts&explaintext=1`
/// at the revision above.
pub const ARTICLE_URL: &str =
    "https://en.wikipedia.org/w/index.php?title=Atlantic_Ocean&oldid=1381231733";
/// The query that produced the pinned bytes — the article page URL returns HTML.
/// The revision is pinned, but the *extract* is not: the endpoint renders the
/// page's current text, so a re-fetch of this same revision on 2026-10-24
/// returned 52 787 bytes (sha256 `426ac83b…`) — the pin plus the two characters
/// of the one edit landed after it, revision 1381660382's `divided in two parts`
/// → `divided into two parts`. Reverting that edit reproduces the pin byte for
/// byte (52 593 characters, sha256 [`ARTICLE_SHA256`]), so a re-run may read the
/// pinned file or re-fetch and undo it, either way checked against the pin.
pub const ARTICLE_FETCH: &str = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&formatversion=2&";
/// The plain-text extract: 52 593 characters (52 785 bytes — the article has
/// multi-byte place names), sha256 over the bytes.
pub const ARTICLE_CHARS: usize = 52_593;
pub const ARTICLE_SHA256: &str = "df0eb1cd6ed0844698e1b63bc34698f2c3591650fede12d4d58e2c48ce0bdc0f";

/// The two cells' row counts, BOS and EOS included — the sequence is BOS, the
/// content, EOS, and nothing else: no padding, no filler.
pub const EMBED_CELLS: [usize; 2] = [256, 8_192];
/// `embed.8192`'s text: the article's first 5 846 words (37 543 bytes), which
/// tokenise to exactly 8 190 content tokens.
pub const CELL_8192_WORDS: usize = 5_846;
pub const CELL_8192_BYTES: usize = 37_543;
pub const CELL_8192_SHA256: &str =
    "40ee7e6f2fa3f530f018d05e3538314ab9c9656b829d4654eb52a7810faaba3c";
/// `embed.256`'s text: 195 words from the article's midpoint sentence (1 251
/// bytes), which tokenise to exactly 254 content tokens.
pub const CELL_256_WORDS: usize = 195;
pub const CELL_256_BYTES: usize = 1_251;
pub const CELL_256_SHA256: &str =
    "727dccf3fdbd6460acd836ef132e0679f2904206f940df20b10c2cbcd47ce960";
/// Where `embed.256`'s excerpt begins.
pub const CELL_256_ANCHOR: &str = "Large quantities of basalt erupted at this time period";

/// The content tokens of a cell: every row but the BOS and the EOS.
#[must_use]
pub fn content_tokens(rows: usize) -> usize {
    rows - 2
}

/// The first `words` space-separated fields of `text`, joined as they were. A
/// **space** is the delimiter, not any whitespace: the article's paragraphs are
/// separated by newlines inside a field, which is how the reference cut was
/// taken and what makes the two pinned byte counts reproducible.
#[must_use]
pub fn words_prefix(text: &str, words: usize) -> &str {
    let mut seen = 0usize;
    for (i, ch) in text.char_indices() {
        if ch == ' ' {
            seen += 1;
            if seen == words {
                return &text[..i];
            }
        }
    }
    text
}

/// The `rows`-row cell's text, cut from the article and checked against its own
/// pin: `embed.8192` is the article's prefix, `embed.256` the excerpt that
/// begins at [`CELL_256_ANCHOR`]. A text that does not match is an error naming
/// the figure that differs — the cells measure a pinned input, not whatever the
/// file happens to hold.
pub fn cell_text(article: &str, rows: usize) -> anyhow::Result<String> {
    let (words, bytes, sha, from) = match rows {
        8_192 => (CELL_8192_WORDS, CELL_8192_BYTES, CELL_8192_SHA256, 0),
        256 => {
            let at = article
                .find(CELL_256_ANCHOR)
                .ok_or_else(|| anyhow::anyhow!("the article does not hold `{CELL_256_ANCHOR}`"))?;
            (CELL_256_WORDS, CELL_256_BYTES, CELL_256_SHA256, at)
        }
        other => anyhow::bail!("no embedding cell of {other} rows — the cells are {EMBED_CELLS:?}"),
    };
    let text = words_prefix(&article[from..], words).to_string();
    if text.len() != bytes {
        anyhow::bail!(
            "cell {rows}: the cut is {} bytes, the pin says {bytes}",
            text.len()
        );
    }
    let got = sha256_hex(text.as_bytes());
    if got != sha {
        anyhow::bail!("cell {rows}: sha256 {got} is not the pinned {sha}");
    }
    Ok(text)
}

/// Whether the run's voice input is the clip the speak witness was recorded
/// with. Both generative cells pin against it and skip their own pin when it
/// differs, so the rule is stated here once: an unreadable voice counts as the
/// pinned one, and the pin then decides on the digest it exists for.
pub fn voice_is_pinned() -> bool {
    crate::loader::voice_wav()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .is_none_or(|bytes| sha256_hex(&bytes) == SPEAK_VOICE_SHA256)
}

/// The article the cells are cut from, checked against [`ARTICLE_SHA256`] — the
/// file's own name and the mismatch are both named when it fails, because the
/// text is an input the checkout does not carry. A file fetched today rather than
/// the pinned one is the expected failure, not a corrupt file: the extract
/// endpoint is not byte-stable ([`ARTICLE_FETCH`]).
pub fn article() -> anyhow::Result<String> {
    let path = crate::loader::embed_text()?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("cannot read `{}`: {e}", path.display()))?;
    let chars = text.chars().count();
    if chars != ARTICLE_CHARS || sha256_hex(text.as_bytes()) != ARTICLE_SHA256 {
        anyhow::bail!(
            "`{}` is not the pinned extract ({chars} characters, sha256 {}, pinned \
             {ARTICLE_CHARS} characters, sha256 {ARTICLE_SHA256}) — the extract endpoint renders \
             the page's current text, so a fresh fetch of the same revision is longer by the one \
             edit landed after it (revision 1381660382); undo that edit or read the pinned file",
            path.display(),
            sha256_hex(text.as_bytes()),
        );
    }
    Ok(text)
}

/// Print the recognition witness against the pins. `false` on any difference.
pub fn check_listen(
    frames: usize,
    prompt_rows: usize,
    generated: usize,
    answer_digest: u64,
    transcript: &str,
) -> bool {
    let hash = sha256_hex(transcript.as_bytes());
    let same = frames == LISTEN_FRAMES
        && prompt_rows == LISTEN_PROMPT_ROWS
        && generated == LISTEN_GENERATED
        && answer_digest == LISTEN_ANSWER_DIGEST
        && hash == LISTEN_TRANSCRIPT_SHA256;
    println!(
        "witness listen frames={frames}/{LISTEN_FRAMES} prompt={prompt_rows}/{LISTEN_PROMPT_ROWS} \
         generated={generated}/{LISTEN_GENERATED} answer_digest={answer_digest:016x}/\
         {LISTEN_ANSWER_DIGEST:016x} transcript_sha256={hash}/{} equal={same}",
        LISTEN_TRANSCRIPT_SHA256
    );
    if !same {
        println!("witness listen — the answer is not the pinned one:");
        println!("  produced: {transcript:?}");
        println!("  pinned:   {LISTEN_TRANSCRIPT:?}");
    }
    same
}

/// Print the synthesis witness: the waveform hash and its sample count.
pub fn check_speak(rows: usize, samples: usize, wave_digest: u64, prompt_digest: u64) -> bool {
    let same = rows == SPEAK_ROWS
        && samples == SPEAK_SAMPLES
        && wave_digest == SPEAK_WAVE_DIGEST
        && prompt_digest == SPEAK_PROMPT_DIGEST;
    println!(
        "witness speak rows={rows}/{SPEAK_ROWS} samples={samples}/{SPEAK_SAMPLES} \
         waveform_digest={wave_digest:016x}/{SPEAK_WAVE_DIGEST:016x} \
         prompt_digest={prompt_digest:016x}/{SPEAK_PROMPT_DIGEST:016x} equal={same}"
    );
    same
}

/// Print the round-trip witness against the pin: what the recogniser heard from
/// the synthesis waveform, and whether it is the pinned answer. The verdict line
/// carries the two transcripts' SHA-256 the way [`check_listen`] carries its own,
/// and the texts print only on a mismatch. `wave=raw` states which side of
/// `prepare_speech` the gate ran on; a mismatch is a failed cell, not a diagnostic.
pub fn check_speak_roundtrip(heard: &str) -> bool {
    let same = heard == SPEAK_TRANSCRIPT;
    let hash = sha256_hex(heard.as_bytes());
    println!(
        "witness speak-roundtrip wave=raw transcript_sha256={hash}/{} equal={same}",
        sha256_hex(SPEAK_TRANSCRIPT.as_bytes())
    );
    if !same {
        println!("witness speak-roundtrip — the recogniser heard a different answer:");
        println!("  heard:  {heard:?}");
        println!("  pinned: {SPEAK_TRANSCRIPT:?}");
    }
    same
}
