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
