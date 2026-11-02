// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The answers the two generative surfaces must keep producing. Recognition's
//! three stand rows are recorded as the SHA-256 of the regions the product call
//! returned, printed beside the run's own. Synthesis is recorded by its words: a
//! render is as long as its text, so no whole waveform is pinned, and the one hard
//! gate is the round trip — the hearing must reach the text's closing words, in
//! order, at its end, and must not say more of it than a quarter again
//! ([`check_speak_roundtrip`]).
//!
//! Taken from a plain run of the stand's rows, on the models the run loads, with no
//! `RLX_*` switch set: the instrumented runs are not witnesses, because a switch may
//! change the scheduling shape.

use std::fmt;

/// The prompt rows one recognition piece is compiled at: a 640-frame rung, 80
/// audio rows and the 15 text ones. The shape a prompt graph is built at, not a
/// reading's total.
pub const LISTEN_PIECE_PROMPT_ROWS: usize = 95;

// The pinned digests in this file are a **witness, not a gate** (owner-directed):
// a hash that moves because the real outputs moved by a rounding is recorded and
// reported — with what moved and by how much — and the run carries on, as long
// as the recognised transcript of the synthesised wave and the speech's
// intelligibility and range are intact. What is hard: no non-finite values and no
// clipping. Do not roll back a measured win because a digest moved in its last
// digits, and do not spend a round chasing one back.
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
/// later window of [`crate::stand::RECOGNITION_RECORDING`] — the pinned corpus
/// recording the row reads, at `WAKE_OFFSET` and `WAKE_REFERENCE_OFFSET` of it. A
/// witness, not a gate.
///
/// The two windows go through the encoder in one batched pass, and this reading
/// is that pass's. It is the one-window pass's to the fourth decimal: a window's mel,
/// its conv rows and its attention are its own at any batch, which the row's own
/// `neighbour_free` reading holds — the first window's embedding is bit-identical
/// when its neighbour changes.
pub const STAND_WAKE_SCORE: f32 = 0.911_153;

/// The text's closing words the round trip demands, in order, at the end of the
/// hearing: four, the shortest run that still proves the ending. The recogniser is not
/// accurate enough for a word-by-word comparison — three quarters of the long
/// paragraph's words come back in order, whole clauses replaced by similar-sounding
/// ones — so every word demanded is another chance to go red on a render that is fine,
/// and four to eight of the text's last words are heard where ten are not.
pub const SPEAK_TAIL_WORDS: usize = 4;
/// How much longer than the text a hearing may be, in words of it: a quarter more. The
/// good render's hearing is 191 words for the text's 201, so the margin is sixty words of
/// headroom over the mishearings; the case it catches, and the only rule that does, is
/// the render that says its text more than once — every phrase at ×2, the ending still
/// reached, and twice the words.
pub const SPEAK_LONGER_BY_PERCENT: usize = 25;

/// The words of `text` with case and punctuation aside — what the recogniser's own
/// spelling does to a spoken sentence, and what [`round_trip`] compares.
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

/// What the round trip's rules make of one hearing of a rendered text. The rules are
/// asked in this order — the ending, the length, the repetition — so a hearing that
/// breaks several reports the first, and a control built to break one names that one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundTrip {
    /// The hearing ends with the text's closing words and stays inside the text's own
    /// bounds: a working render, whatever the recogniser misheard along the way.
    Holds,
    /// The hearing does not end with the text's last [`SPEAK_TAIL_WORDS`] words, in
    /// order: a render cut short, or one that spent its length on a loop and never got
    /// there.
    MissedEnding,
    /// The hearing says `heard` words where the text says `text` — more than
    /// [`SPEAK_LONGER_BY_PERCENT`] more, which is a render that said its text more than
    /// once.
    RanLong { heard: usize, text: usize },
    /// One [`crate::transcribe::REPEAT_GRAM`]-word phrase is said `said` times over the
    /// whole hearing, past the [`crate::transcribe::REPEAT_LIMIT`] the product's own
    /// guards allow a chunk of it.
    Looped { said: usize },
}

impl RoundTrip {
    pub fn holds(&self) -> bool {
        *self == Self::Holds
    }
}

impl fmt::Display for RoundTrip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::transcribe::{REPEAT_GRAM, REPEAT_LIMIT};
        match self {
            Self::Holds => write!(
                f,
                "reaches the text's closing {SPEAK_TAIL_WORDS} words within \
                 {SPEAK_LONGER_BY_PERCENT}% of its length, no phrase past ×{REPEAT_LIMIT}"
            ),
            Self::MissedEnding => write!(
                f,
                "does not end with the text's closing {SPEAK_TAIL_WORDS} words, in order"
            ),
            Self::RanLong { heard, text } => write!(
                f,
                "says {heard} words where the text says {text}: more than \
                 {SPEAK_LONGER_BY_PERCENT}% longer"
            ),
            Self::Looped { said } => write!(
                f,
                "says one {REPEAT_GRAM}-word phrase {said} times over, past the \
                 ×{REPEAT_LIMIT} the guards allow a chunk"
            ),
        }
    }
}

/// The round trip's rules over one hearing of a rendered text, case and punctuation
/// aside: does it reach the text's closing words, and does it stay inside the text's
/// bounds? Accuracy is not a rule — the recogniser mishears a quarter of the words on a
/// correct render — and neither is any repetition at all, which it also produces; the
/// ending is what a cut or a looped render destroys, and the bounds are the product's
/// own repetition limit plus the length the text itself sets.
pub fn round_trip(heard: &str, spoken: &str) -> RoundTrip {
    use crate::transcribe::{REPEAT_LIMIT, repeat_count};
    let (heard, spoken) = (words(heard), words(spoken));
    let tail = SPEAK_TAIL_WORDS.min(spoken.len());
    if tail == 0
        || heard.len() < tail
        || heard[heard.len() - tail..] != spoken[spoken.len() - tail..]
    {
        return RoundTrip::MissedEnding;
    }
    let allowed = spoken.len() + spoken.len() * SPEAK_LONGER_BY_PERCENT / 100;
    if heard.len() > allowed {
        return RoundTrip::RanLong {
            heard: heard.len(),
            text: spoken.len(),
        };
    }
    match repeat_count(&heard) {
        Some(said) if said > REPEAT_LIMIT => RoundTrip::Looped { said },
        _ => RoundTrip::Holds,
    }
}

/// Print the round trip's reading: what the recogniser heard from the render and which
/// rule it broke, with the figures behind them, and the hearing itself. The judgement is
/// on words, not on a hash — the waveform's bits are the sampler's, its words are the
/// product's, and the recogniser's mishearings are its own. A red is diagnosable from
/// what this prints: the hearing, the text, and the rule that fired.
pub fn check_speak_roundtrip(heard: &str, spoken: &str) -> bool {
    use crate::transcribe::{REPEAT_GRAM, REPEAT_LIMIT, repeat_count};
    let verdict = round_trip(heard, spoken);
    let heard_words = words(heard);
    println!(
        "witness speak-roundtrip holds={} heard_words={} spoken_words={} \
         worst_{REPEAT_GRAM}-gram=×{} limit=×{REPEAT_LIMIT} tail_words={SPEAK_TAIL_WORDS} \
         longer_by={SPEAK_LONGER_BY_PERCENT}% — {verdict}",
        verdict.holds(),
        heard_words.len(),
        words(spoken).len(),
        repeat_count(&heard_words).unwrap_or(0),
    );
    println!("  heard: {heard:?}");
    if !verdict.holds() {
        println!("  spoken: {spoken:?}");
    }
    verdict.holds()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The round trip's three rules and the order they are asked in, on a text long
    /// enough that a quarter more words than it can hold a phrase said four times: a
    /// misheard middle passes, a cut or a wrong opening ending short fails, a phrase past
    /// the guards' limit is a loop, and the text said twice runs long.
    #[test]
    fn the_round_trip_is_judged_on_the_texts_ending() {
        // Sixty-four distinct words: the tail's four are `word61`–`word64`, and a quarter
        // more words than the text's sixty-four is sixteen.
        let spoken = (1..=64)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tail = "word61 word62 word63 word64";
        // The recogniser's own mishearings in the middle decide nothing.
        assert_eq!(
            round_trip(&format!("something else entirely {tail}"), &spoken),
            RoundTrip::Holds
        );
        assert_eq!(round_trip("", &spoken), RoundTrip::MissedEnding);
        assert_eq!(
            round_trip("word61 word62 word63", &spoken),
            RoundTrip::MissedEnding
        );
        // The accepted trade: a render that starts wrong and ends right passes. The
        // opening is printed for a person; the ending is what a cut destroys.
        assert_eq!(
            round_trip(&format!("some other opening {tail}"), &spoken),
            RoundTrip::Holds
        );
        // One phrase five times over — the four at the head, the text's own at its end —
        // inside the length the text allows: the guards' limit is what fails it.
        assert_eq!(
            round_trip(&format!("{tail} {tail} {tail} {tail} {spoken}"), &spoken),
            RoundTrip::Looped { said: 5 }
        );
        // At the guards' own limit, not past it, is a phrase said the way the text says
        // it: no new bar.
        assert_eq!(
            round_trip(&format!("{tail} {tail} {spoken}"), &spoken),
            RoundTrip::Holds
        );
        // The text said twice: every phrase at ×2, the ending reached, and twice the
        // words — the length rule is the one that catches it.
        assert_eq!(
            round_trip(&format!("{spoken} {spoken}"), &spoken),
            RoundTrip::RanLong {
                heard: 128,
                text: 64
            }
        );
        // The ending is asked first: a hearing that both runs long and stops short is a
        // missed ending, not a runaway.
        let long_and_short = format!("{spoken} {spoken}");
        assert_eq!(
            round_trip(long_and_short.trim_end_matches(" word64"), &spoken),
            RoundTrip::MissedEnding
        );
    }
}
