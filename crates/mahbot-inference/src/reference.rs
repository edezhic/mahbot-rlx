// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The answers the two generative surfaces must keep producing. Recognition is
//! recorded by its words — the transcript should still read as [`LISTEN_TRANSCRIPT`],
//! with at most [`LISTEN_MOVED_WORDS`] of them different — and its digest, its token
//! count and every moved word print as witnesses. Synthesis is recorded by its words
//! too: a render is as long as its text, so no whole waveform is pinned, and the one
//! hard gate is the round trip — the recogniser must hear the text the render was
//! asked to speak ([`check_speak_roundtrip`]).
//!
//! Taken from a plain run of the stand's rows, on the models the run loads, with no
//! `RLX_*` switch set: the instrumented runs are not witnesses, because a switch may
//! change the scheduling shape.

use sha2::{Digest, Sha256};

/// Recognition: the clip is the recorded 17.46 s Russian reference, 1 746 mel
/// frames → a 234-row prompt (3 head + 6 text + 219 audio + 6 tail).
pub const LISTEN_FRAMES: usize = 1_746;
pub const LISTEN_PROMPT_ROWS: usize = 234;
pub const LISTEN_GENERATED: usize = 65;
// The pinned digests in this file are a **witness, not a gate** (owner-directed):
// a hash that moves because the real outputs moved by a rounding is recorded and
// reported — with what moved and by how much — and the run carries on, as long
// as the recognised transcript of the synthesised wave and the speech's
// intelligibility and range are intact. What is hard: no non-finite values and no
// clipping. Do not roll back a measured win because a digest moved in its last
// digits, and do not spend a round chasing one back.
/// `digest(generated ids ++ next ++ eos flag ++ written)` — the row's own answer
/// digest, printed as `digest=`.
pub const LISTEN_ANSWER_DIGEST: u64 = 0x2f93_dfa1_d7cd_eeb1;
/// SHA-256 of the decoded transcript, in UTF-8.
pub const LISTEN_TRANSCRIPT_SHA256: &str =
    "522e47e1f6ed20ca69ad638df1b2b44ed03ef721e349d6ffacea4e13f7c18d3f";
pub const LISTEN_TRANSCRIPT: &str = "обычные объявления метро делались только на каталонском языке, но не забывали переводить объявления в другие языки, например, итальянский, испанский, английский, французский, итальянский";
/// Words of [`LISTEN_TRANSCRIPT`] a run may differ in and still read as it: one
/// moved token lands in one word, and two such moves are still the same sentence.
/// The word count itself must match — a move across a space merges or splits a
/// word, and that is a different sentence rather than a moved word.
pub const LISTEN_MOVED_WORDS: usize = 2;

/// The stand's three recognition rows — a 10 s and a 30 s prefix of the pinned
/// recording and the whole of it — as the SHA-256 of the regions the product call
/// returned: each region's bounds, its read flag and its transcript or label. A
/// witness, not a gate: the row prints the two side by side.
pub const STAND_RECOGNITION_SHORT_SHA256: &str =
    "9f86db8f384e0534b912fd5ccdea9fca7e98b43faadf61ad53cd44be5afa33f3";
pub const STAND_RECOGNITION_TYPICAL_SHA256: &str =
    "d85b1f85aa309a811ba6a7d7654ef5793c49e7b7166a1ce187a8ed743a98ef28";
pub const STAND_RECOGNITION_LONG_SHA256: &str =
    "9a306f23337517fc774856f9350b3f115b0252cae9a8c3eab39a56503fdcf9ce";

/// The stand's two synthesis rows at their draw seed: `digest(&waveform)` and
/// the waveform's length. A witness, not a gate.
pub const STAND_SYNTH_SHORT_DIGEST: u64 = 0x2b28_d437_6a94_7067;
pub const STAND_SYNTH_SHORT_SAMPLES: usize = 44_160;
pub const STAND_SYNTH_TYPICAL_DIGEST: u64 = 0x213d_9b26_4c44_bebb;
pub const STAND_SYNTH_TYPICAL_SAMPLES: usize = 172_800;

/// The stand's wake row: the cosine the pair pass's first window answers against a
/// later window of the same recording. A witness, not a gate.
///
/// The two windows now go through the encoder in one batched pass, and this reading
/// is that pass's. It is the one-window pass's to the fourth decimal: a window's mel,
/// its conv rows and its attention are its own at any batch, which the row's own
/// `neighbour_free` reading holds — the first window's embedding is bit-identical
/// when its neighbour changes.
pub const STAND_WAKE_SCORE: f32 = 0.8846;

/// The voice the synthesis witness was recorded with. `MAHBOT_VOICE_WAV` names
/// the voice input, so a run that points it at another file expects another
/// waveform and skips the pin.
pub const SPEAK_VOICE_SHA256: &str =
    "0ca6c18b9189381dcdad83d26814f73b3f4d2d26dea6892371c2177d30e72ace";
/// How many of the spoken text's opening words the round trip demands: enough that a
/// waveform of noise or a silent one cannot pass, short enough that the recogniser's
/// own reading of the rest of a long render decides nothing.
const SPEAK_OPENING_WORDS: usize = 4;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    crate::measurements::hex(&h.finalize())
}

/// Whether the run's voice input is the clip the speak witness was recorded
/// with. The round trip is judged with it — another voice renders other words, which
/// the check cannot judge — and the pinned-clip check compares the listen witness
/// only when it holds, so the rule is stated here once: an unreadable voice counts as
/// the pinned one, and the pin then decides on the digest it exists for.
pub fn voice_is_pinned() -> bool {
    crate::loader::voice_wav()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .is_none_or(|bytes| sha256_hex(&bytes) == SPEAK_VOICE_SHA256)
}

/// One run's transcript split against the pin's own words — split once, so the
/// verdict, the moved-word lines and the word counts `check_listen` prints cannot
/// disagree. The rule is the sentence, not the hash: a token moves on a last-bit
/// rounding in the multiply, which is exactly what a measured change is allowed to
/// do, and a pinned SHA-256 failed the check for it.
struct Words<'a> {
    /// `(index, produced, pinned)` for each word that differs, pairwise.
    moved: Vec<(usize, &'a str, &'static str)>,
    produced: usize,
    pinned: usize,
    /// Same word count and at most [`LISTEN_MOVED_WORDS`] words different.
    reads_as_pinned: bool,
}

impl<'a> Words<'a> {
    fn against_pin(transcript: &'a str) -> Self {
        let got: Vec<&'a str> = transcript.split_whitespace().collect();
        let want: Vec<&'static str> = LISTEN_TRANSCRIPT.split_whitespace().collect();
        let moved = got
            .iter()
            .zip(&want)
            .enumerate()
            .filter(|(_, (g, w))| g != w)
            .map(|(i, (g, w))| (i, *g, *w))
            .collect::<Vec<_>>();
        let reads_as_pinned = got.len() == want.len() && moved.len() <= LISTEN_MOVED_WORDS;
        Self {
            moved,
            produced: got.len(),
            pinned: want.len(),
            reads_as_pinned,
        }
    }
}

/// The scalar fields a run's witness moved on, one entry each. The transcript's word
/// count is one of them: [`Words`] compares pairwise over the shared prefix, so a
/// transcript that ends early or runs long moves none of its words.
fn moved_fields(
    frames: usize,
    prompt_rows: usize,
    generated: usize,
    answer_digest: u64,
    words: &Words,
) -> Vec<String> {
    let mut moved = Vec::new();
    if frames != LISTEN_FRAMES {
        moved.push(format!("frames {frames} (pinned {LISTEN_FRAMES})"));
    }
    if prompt_rows != LISTEN_PROMPT_ROWS {
        moved.push(format!(
            "prompt {prompt_rows} (pinned {LISTEN_PROMPT_ROWS})"
        ));
    }
    if generated != LISTEN_GENERATED {
        moved.push(format!("generated {generated} (pinned {LISTEN_GENERATED})"));
    }
    if answer_digest != LISTEN_ANSWER_DIGEST {
        moved.push(format!(
            "answer_digest {answer_digest:016x} (pinned {LISTEN_ANSWER_DIGEST:016x})"
        ));
    }
    if words.produced != words.pinned {
        moved.push(format!(
            "words {} (pinned {})",
            words.produced, words.pinned
        ));
    }
    moved
}

/// The `witness listen — moved:` headline: the scalar fields that moved, else the
/// words that did, else — nothing having moved — the text's own bytes, which is the
/// only case where the headline can claim every word is at its pin.
fn moved_headline(moved: &[String], words: &Words) -> String {
    if !moved.is_empty() {
        moved.join(", ")
    } else if !words.moved.is_empty() {
        format!(
            "{} of {} words (listed below)",
            words.moved.len(),
            words.produced
        )
    } else {
        "the transcript's bytes only (every word at its pin)".to_string()
    }
}

/// Print the recognition witness against its pins: every field is compared once
/// here, and the transcript is split once by `Words`. Answers whether the clip's
/// own reading holds and its words still read as the pin — the digest and the
/// generated count print beside them and decide nothing — and prints what moved
/// when it did, so a caller that reports the witness can say which pin moved (the
/// owner-directed rule at the pins below).
pub fn check_listen(
    frames: usize,
    prompt_rows: usize,
    generated: usize,
    answer_digest: u64,
    transcript: &str,
) -> bool {
    let hash = sha256_hex(transcript.as_bytes());
    let words = Words::against_pin(transcript);
    let moved = moved_fields(frames, prompt_rows, generated, answer_digest, &words);
    // The clip's own reading is the check's identity: a run that read the input
    // differently is not this check, whatever it answered.
    let input_is_pinned = frames == LISTEN_FRAMES && prompt_rows == LISTEN_PROMPT_ROWS;
    let all_pinned = moved.is_empty() && hash == LISTEN_TRANSCRIPT_SHA256;
    println!(
        "witness listen frames={frames}/{LISTEN_FRAMES} prompt={prompt_rows}/{LISTEN_PROMPT_ROWS} \
         generated={generated}/{LISTEN_GENERATED} answer_digest={answer_digest:016x}/\
         {LISTEN_ANSWER_DIGEST:016x} transcript_sha256={hash}/{} all_pinned={all_pinned} \
         reads_as_pinned={}",
        LISTEN_TRANSCRIPT_SHA256, words.reads_as_pinned
    );
    if !all_pinned {
        println!("witness listen — moved: {}", moved_headline(&moved, &words));
        for (i, got, want) in &words.moved {
            println!("  word {i}: produced {got:?}, pinned {want:?}");
        }
    }
    if !words.reads_as_pinned {
        println!("  produced: {transcript:?}");
        println!("  pinned:   {LISTEN_TRANSCRIPT:?}");
    }
    words.reads_as_pinned && input_is_pinned
}

/// The words of `text` with case and punctuation aside — what the recogniser's own
/// spelling does to a spoken sentence, and what [`check_speak_roundtrip`] compares.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Whether what the recogniser heard **opens with the text the render was asked to
/// speak** — [`SPEAK_OPENING_WORDS`] of them, case and punctuation aside.
fn hears_opening(heard: &str, spoken: &str) -> bool {
    let (heard, spoken) = (words(heard), words(spoken));
    let count = SPEAK_OPENING_WORDS.min(spoken.len());
    count > 0 && heard.len() >= count && heard[..count] == spoken[..count]
}

/// Print the round-trip witness: what the recogniser heard from the synthesis
/// waveform, and whether it opens with the text that was spoken. The judgement is on
/// words, not on a hash — the waveform's bits are the sampler's, its words are the
/// product's, and the recogniser's reading of a long render is its own. The hearing
/// prints when it does not open with the text, so a mismatch says what it heard
/// instead. `wave=raw` states which side of the recognition side's conditioning the
/// check ran on; a mismatch is a failed check, not a diagnostic.
pub fn check_speak_roundtrip(heard: &str, spoken: &str) -> bool {
    let holds = hears_opening(heard, spoken);
    println!(
        "witness speak-roundtrip wave=raw opening={holds} heard_words={} spoken_words={} \
         opening_words={SPEAK_OPENING_WORDS}",
        words(heard).len(),
        words(spoken).len(),
    );
    if !holds {
        println!("witness speak-roundtrip — the recogniser did not hear the text's opening:");
        println!("  heard:  {heard:?}");
        println!("  spoken: {spoken:?}");
    }
    holds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Words::against_pin`]'s verdict for one transcript.
    fn reads_as_pinned(transcript: &str) -> bool {
        Words::against_pin(transcript).reads_as_pinned
    }

    /// The listen witness' rule's own boundaries: the pinned sentence and an
    /// empty or differently-worded one. A moved word and a truncated sentence are
    /// the check's cases, below.
    #[test]
    fn a_sentence_off_its_words_is_not_the_pinned_one() {
        assert!(reads_as_pinned(LISTEN_TRANSCRIPT));
        assert!(!reads_as_pinned(""));
        assert!(!reads_as_pinned(&LISTEN_TRANSCRIPT.replace(
            "итальянский, испанский, английский",
            "немецкий, шведский, датский"
        )));
    }

    /// [`check_listen`]'s verdict composition — which pins decide: a moved
    /// digest, generated count, word or spacing is a record the check prints and
    /// carries on with, and only the clip's own reading and a sentence off its words
    /// leave the pins.
    #[test]
    fn the_witness_verdict_says_which_pins_held() {
        assert!(check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            LISTEN_TRANSCRIPT
        ));
        assert!(check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST ^ 1,
            LISTEN_TRANSCRIPT
        ));
        assert!(check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED + 1,
            LISTEN_ANSWER_DIGEST,
            LISTEN_TRANSCRIPT
        ));
        assert!(check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            &LISTEN_TRANSCRIPT.replace("на каталонском языке", "на каталонской языке")
        ));
        // The text's own bytes moving while every word holds its place — spacing —
        // is a record too, not a failure.
        assert!(check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            &LISTEN_TRANSCRIPT.replacen(", ", ",  ", 1)
        ));
        // The clip's own reading is the check's identity, not its answer.
        assert!(!check_listen(
            LISTEN_FRAMES + 1,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            LISTEN_TRANSCRIPT
        ));
        assert!(!check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS + 1,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            LISTEN_TRANSCRIPT
        ));
        // A truncated sentence is off its words whatever the witness says.
        assert!(!check_listen(
            LISTEN_FRAMES,
            LISTEN_PROMPT_ROWS,
            LISTEN_GENERATED,
            LISTEN_ANSWER_DIGEST,
            "обычные объявления метро"
        ));
    }

    /// The headline [`check_listen`] prints for the three cases it covers: a moved
    /// word, a count drift and the spacing-only transcript. The verdict is the same
    /// in the first two, so what the check says is what this pins — the reassurance
    /// may appear only when nothing moved at all.
    #[test]
    fn the_moved_headline_names_every_case_it_covers() {
        fn headline(transcript: &str) -> String {
            let words = Words::against_pin(transcript);
            let moved = moved_fields(
                LISTEN_FRAMES,
                LISTEN_PROMPT_ROWS,
                LISTEN_GENERATED,
                LISTEN_ANSWER_DIGEST,
                &words,
            );
            moved_headline(&moved, &words)
        }
        assert_eq!(
            headline(&LISTEN_TRANSCRIPT.replacen(", ", ",  ", 1)),
            "the transcript's bytes only (every word at its pin)"
        );
        let one_word = LISTEN_TRANSCRIPT.replace("на каталонском языке", "на каталонской языке");
        assert_eq!(headline(&one_word), "1 of 22 words (listed below)");
        let truncated = "обычные объявления метро";
        assert_eq!(headline(truncated), "words 3 (pinned 22)");
    }

    /// The round trip's rule: the spoken text's opening is what decides, case and
    /// punctuation aside, and a hearing that stops short of it or starts somewhere else
    /// does not pass. The printing wrapper around it is not what is under test.
    #[test]
    fn the_round_trip_is_judged_on_the_texts_opening() {
        let spoken = "Call me Ishmael. Some years ago, never mind how long precisely";
        assert!(hears_opening(
            "Call me Ishmael, some years ago, and then it went on",
            spoken
        ));
        assert!(!hears_opening("", spoken));
        assert!(!hears_opening("anything at all", ""));
        assert!(!hears_opening("Call me", spoken));
        assert!(!hears_opening("Ishmael, call me some years ago", spoken));
    }
}
