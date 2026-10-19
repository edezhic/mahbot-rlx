// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bench's fixed material: phrase lists, seed bands, the silence and noise
//! matrices, and the pinned real-audio subset. Everything here is a constant so
//! two runs differ only in the models they were given.
//!
//! The seed bands are disjoint by construction: enrolment 100–109, owner
//! negatives 9000–9029, confusable negatives 1000+, unrelated negatives 2000+,
//! detection confusables 800/810, detection unrelated 900/910, held-out recall
//! 3000–3039, command-plus-speech 3100–3109, phrase-in-passing 3200–3207, and
//! the enrolment conditioning floor at
//! [`audio::CONDITIONING_NOISE_SEED_BASE`]` + clip index`. A clip from one band
//! is never used in another set, so no clip is both calibration material and
//! measurement material.

use super::audio::{self, NoiseGenerator};

/// The fixed wake phrase every run synthesizes and tests.
pub(crate) const WAKE_PHRASE: &str = "hey mahbot";

/// Enrolment variants: ten conditioned renderings of the one enrolled voice.
pub(crate) const NUM_ENROLLMENT_VARIANTS: usize = 10;
/// Seeds `ENROLLMENT_SEED_BASE..+NUM_ENROLLMENT_VARIANTS`.
pub(crate) const ENROLLMENT_SEED_BASE: u64 = 100;

/// Owner negatives: the enrolled voice saying something that is not the wake
/// phrase. Disjoint from every other phrase list.
pub(crate) const OWNER_NEGATIVE_PHRASES: &[&str] = &[
    "please stop",
    "i am going out",
    "where are my keys",
    "the meeting starts soon",
    "let me check the weather",
    "remind me to call the dentist",
    "what should we have for dinner",
    "the wifi keeps disconnecting",
    "i need to buy new shoes",
    "turn down the music please",
];
pub(crate) const OWNER_NEGATIVE_SEEDS_PER_PHRASE: usize = 3;
pub(crate) const OWNER_NEGATIVE_SEED_BASE: u64 = 9000;
/// The bounded low-SNR cell added per owner-negative clip: brown noise at this
/// SNR on the VAD-gated speech.
pub(crate) const OWNER_NEGATIVE_SNR_DB: f32 = 10.0;

/// Confusable near-misses for the negative pools.
pub(crate) const CONFUSABLE_PHRASES: &[&str] = &[
    // ── Direct phonetic substitutions (wake-word-like) ──
    "hey madbot",
    "hey map bot",
    "day mahbot",
    "hey nab it",
    "hey man",
    "hey mabot",
    "hey mahbott",
    "hey mat",
    "hey max",
    "pay mabot",
    // ── Rhythmic/melodic confusables ──
    "hay map pot",
    "huh mahbot",
    "eh mad bot",
    "hey maybott",
    "they mad bot",
    "haymaker",
    // ── Embedded wake-word sounds ──
    "hey maybe not",
    "play mah jong",
    "hey matter of fact",
    "a day with mahbot",
    // ── Short phonetic near-misses ──
    "madbot",
    "mat bot",
    "bad bot",
    "mad lot",
    "mad pot",
    "med bot",
    "my bot",
    "may bot",
];
/// Seeds per confusable phrase in the calibration pool, and their base.
pub(crate) const CONFUSABLE_SEEDS_PER_PHRASE: usize = 5;
pub(crate) const CONFUSABLE_SEED_BASE: u64 = 1000;

/// Unrelated speech for the negative pools.
pub(crate) const UNRELATED_PHRASES: &[&str] = &[
    // ── Short commands (2-4 words) ──
    "the weather today is sunny",
    "what time is it",
    "one two three four five",
    "hello world",
    "good morning everyone",
    "turn on the lights",
    "play some music",
    "set a timer",
    // ── Medium phrases (5-8 words) ──
    "i need to buy groceries today",
    "can you remind me of my appointment",
    "please send a message to john",
    "what is the capital of france",
    "tell me a joke about programming",
    "how do I get to the airport",
    "the quick brown fox jumps over the lazy dog",
    // ── Long utterances (10+ words) ──
    "according to all known laws of aviation there is no way a bee should be able to fly",
    "the principle of superposition states that a quantum system exists in all its possible states simultaneously",
    // ── Non-English phrases (phonetically distinct from the wake phrase) ──
    "bonjour comment allez vous aujourd hui",
    "buenos días cómo estás",
    "guten morgen wie geht es dir",
];
pub(crate) const UNRELATED_SEEDS_PER_PHRASE: usize = 3;
pub(crate) const UNRELATED_SEED_BASE: u64 = 2000;

/// Recall set: [`HELD_OUT_WAKE_ONLY_CLIPS`] bare wake-phrase renderings.
pub(crate) const HELD_OUT_SEED_BASE: u64 = 3000;
pub(crate) const HELD_OUT_WAKE_ONLY_CLIPS: usize = 40;

/// Recall set: the wake phrase at the start of an utterance, followed by a
/// command — the shape a user actually speaks.
pub(crate) const COMMAND_PLUS_SPEECH_PHRASES: &[&str] = &[
    "hey mahbot turn on the kitchen light",
    "hey mahbot play something quiet please",
    "hey mahbot call my sister back",
    "hey mahbot set an alarm for six",
    "hey mahbot add milk to the list",
    "hey mahbot open yesterdays notes",
    "hey mahbot check the weather tomorrow",
    "hey mahbot tell me the headlines",
    "hey mahbot remind me about dinner",
    "hey mahbot what time is the train",
];
pub(crate) const COMMAND_PLUS_SPEECH_SEED_BASE: u64 = 3100;

/// False-reaction set: the wake phrase mentioned in passing inside a longer
/// utterance. Refusing these is success.
pub(crate) const PHRASE_IN_PASSING_PHRASES: &[&str] = &[
    "so i was telling my brother hey mahbot should wait until tomorrow",
    "the plan for saturday morning is hey mahbot can bring the tickets",
    "when you look at the report hey mahbot thinks the numbers are wrong",
    "the weather has been odd this week and hey mahbot says the garden needs rain",
    "there is a long story about the move but hey mahbot already heard most of it",
    "yesterday we walked by the river and hey mahbot took some pictures",
    "i have been reading about the railways and hey mahbot says it rambles",
    "on my way home i bought batteries and hey mahbot asked for coffee",
];
pub(crate) const PHRASE_IN_PASSING_SEED_BASE: u64 = 3200;

/// Seed bands of the detection negatives: two confusable bands and two
/// unrelated bands, one render per phrase per band.
pub(crate) const CONFUSABLE_BANDS: &[(u64, &str)] = &[(800, "confusable"), (810, "confusable2")];
pub(crate) const UNRELATED_BANDS: &[(u64, &str)] = &[(900, "unrelated"), (910, "unrelated2")];

/// The false-reaction set's silence matrix.
pub(crate) const SILENCE_DURATIONS: &[(&str, usize)] = &[
    ("silence_0_5s", 8_000),
    ("silence_1_0s", 16_000),
    ("silence_2_0s", 32_000),
];

/// Noise audio length in samples (one second at 16 kHz).
pub(crate) const NOISE_LEN: usize = 16_000;

/// The noise seed band of the detection set — the original generators' own
/// seeds, in table order (`42 + index`).
pub(crate) const DETECTION_NOISE_SEED_BASE: u64 = 42;
/// The noise seed band of the calibration set. Deliberately disjoint from the
/// detection band so no waveform is both calibration material and measurement
/// material.
pub(crate) const CALIBRATION_NOISE_SEED_BASE: u64 = 4242;

/// Noise profiles the calibration and detection sets share by *name* but not
/// by waveform: the detection set renders them at
/// [`DETECTION_NOISE_SEED_BASE`]`..+14`, the calibration set at
/// [`CALIBRATION_NOISE_SEED_BASE`]`..+10`.
pub(crate) const NOISE_PROFILES: &[(&str, NoiseGenerator)] = &[
    ("white uniform noise", audio::white_uniform_noise),
    ("white gaussian noise", audio::white_gaussian_noise),
    ("pink noise", audio::pink_noise),
    ("brown noise", audio::brown_noise),
    ("mixed speech+noise", audio::mixed_speech_noise),
    ("blue noise", audio::blue_noise),
    ("violet noise", audio::violet_noise),
    ("low-frequency rumble", audio::low_freq_rumble),
    ("modulated noise", audio::modulated_noise),
    ("high-frequency hiss", audio::high_freq_hiss),
];

/// Detection-only noise profiles: they extend the false-reaction set and never
/// enter the calibration pool.
pub(crate) const NOISE_PROFILES_DETECTION_ONLY: &[(&str, NoiseGenerator)] = &[
    ("impulse burst noise", audio::impulse_burst_noise),
    ("oscillating tone noise", audio::oscillating_tone_noise),
    ("crackle static noise", audio::crackle_static_noise),
    ("pulsed broadband noise", audio::pulsed_broadband_noise),
];

/// The detection set's noise profiles: the training list plus the
/// detection-only additions, in table order.
pub(crate) fn all_noise_profiles() -> impl Iterator<Item = &'static (&'static str, NoiseGenerator)>
{
    NOISE_PROFILES.iter().chain(NOISE_PROFILES_DETECTION_ONLY)
}

/// The false-reaction set's size, built from the components above.
/// [`audio::non_phrase_clips`] asserts the set it builds is exactly this long,
/// so a dropped clip is a build-time failure rather than a smaller denominator.
pub(crate) const EXPECTED_NON_PHRASE_TOTAL: usize = CONFUSABLE_BANDS.len()
    * CONFUSABLE_PHRASES.len()
    + UNRELATED_BANDS.len() * UNRELATED_PHRASES.len()
    + SILENCE_DURATIONS.len()
    + NOISE_PROFILES.len()
    + NOISE_PROFILES_DETECTION_ONLY.len()
    + PHRASE_IN_PASSING_PHRASES.len();

/// Where the real-audio corpus came from, as its manifest records it.
pub(crate) const REAL_AUDIO_REPO: &str = "alexwengg/musan_mini";
pub(crate) const REAL_AUDIO_REVISION: &str = "54ce99dd44add93cec39c4d37bdcc59c7870a849";

/// The pinned real-audio subset: 18 speech files + 15 noise files, in this
/// fixed order, each with its SHA-256 and byte size. Selection at pin time: the
/// longest librivox speech files, all us-gov speech files, and the longest
/// free-sound and sound-bible noise clips.
pub(crate) const REAL_AUDIO_FILES: &[(&str, &str, u64)] = &[
    // 6 longest librivox speech files (≈31.2 min)
    (
        "speech/librivox/speech-librivox-0027.wav",
        "82448930a85372948cc65ffa80bad9639dff73189c6f868e2a80b8670481dd3d",
        10_176_514,
    ),
    (
        "speech/librivox/speech-librivox-0051.wav",
        "e21daa2ee0869f7dd53dfe27e8c43b711557dc0d98a84697807f7accfe4482f5",
        10_097_938,
    ),
    (
        "speech/librivox/speech-librivox-0093.wav",
        "17eb2f15ceecb574a8b5b8bf0b74052be272a6fa6d62c65cb384c3a9fd470156",
        10_052_798,
    ),
    (
        "speech/librivox/speech-librivox-0075.wav",
        "140633595fc5213f93bb309a7cf91c349ee674dc7666cd8d2bb4da489a6482cc",
        10_008_494,
    ),
    (
        "speech/librivox/speech-librivox-0103.wav",
        "a61c522b6ee583f3dbae90521bb3a2b038b07c1b303cf9bbae61877df39fe9c2",
        9_830_444,
    ),
    (
        "speech/librivox/speech-librivox-0088.wav",
        "f90a7b3089b66b401bc5008268759e0d18fcba93912b4beefa6c943f1eeb7882",
        9_735_986,
    ),
    // All 12 us-gov speech files (≈25.0 min)
    (
        "speech/us-gov/speech-us-gov-0252.wav",
        "75953157cfe7d5e485d1b023e2353ded908a55144a1df2b2b193163016791e73",
        7_409_612,
    ),
    (
        "speech/us-gov/speech-us-gov-0147.wav",
        "0583526a96cec8bc775b10bc58303f2e5300ff5ed026bbc62c0baefd1dc1520d",
        6_908_072,
    ),
    (
        "speech/us-gov/speech-us-gov-0082.wav",
        "41522bf9652a065f2ebcf883ac8c3b700608607962051fd11a88b8794e9e7d80",
        6_878_816,
    ),
    (
        "speech/us-gov/speech-us-gov-0159.wav",
        "86c3095b47c4696fae943f5975be9bda8b35594832625c0ad63b24a3824620f8",
        5_564_752,
    ),
    (
        "speech/us-gov/speech-us-gov-0067.wav",
        "1ad8de2e00e0399d063bf9a8d1295c1be6bcf05265d3e647e54ba3a7edc5b92e",
        5_048_156,
    ),
    (
        "speech/us-gov/speech-us-gov-0250.wav",
        "cfa92e2fcc91915309ada658abdcba58f35e21b35ab44dbfeeeb0777d8fd8673",
        4_035_856,
    ),
    (
        "speech/us-gov/speech-us-gov-0210.wav",
        "57a9bf8b75b39b858002b3dfbb0af9c8dbeb2d8c88a26479d6ac3dcedad8d69f",
        3_963_132,
    ),
    (
        "speech/us-gov/speech-us-gov-0229.wav",
        "1508a44f0d4043b48354ce0ee330218124f8c7d5b0f11200c771a567def8d658",
        3_214_956,
    ),
    (
        "speech/us-gov/speech-us-gov-0201.wav",
        "9cde92d083a0a02da33d1d1d1576a18868bc33226ad85529bf7ecacf7a921b18",
        1_856_620,
    ),
    (
        "speech/us-gov/speech-us-gov-0130.wav",
        "5de19601dcc136b6ba7b29f0c7c65a6b24269ff33b5317f8acb51f88f38f6cce",
        1_705_316,
    ),
    (
        "speech/us-gov/speech-us-gov-0005.wav",
        "41d7f4388f673d4f6e4f1e71428dbc6cc138ae665b66de0f050762b701c62ca6",
        857_132,
    ),
    (
        "speech/us-gov/speech-us-gov-0109.wav",
        "020f6cda0c2774eb6c2c9fea5afc67f1e7734a2751c9484039251ca13c3ccfec",
        629_492,
    ),
    // 10 longest free-sound noise clips (≈5.3 min)
    (
        "noise/free-sound/noise-free-sound-0640.wav",
        "5dd4086007f68d7ecf5d50edf8016cb15a73397afa8c71e418ddd5a56337d858",
        1_049_122,
    ),
    (
        "noise/free-sound/noise-free-sound-0184.wav",
        "0dbf22f9fe47112820bf783dc031a72f46772bb4fe7f170de4f493e6fc417295",
        1_037_080,
    ),
    (
        "noise/free-sound/noise-free-sound-0761.wav",
        "6cbcde6269a37a98658866bdfbeddbb4d4db745b8065e0670dec0c0cf6762d77",
        1_036_814,
    ),
    (
        "noise/free-sound/noise-free-sound-0012.wav",
        "b81a9ef231e1fb5446773517b84134d711a6b90ce0b19fc93edae6cd901ea0ca",
        1_029_044,
    ),
    (
        "noise/free-sound/noise-free-sound-0605.wav",
        "981d1bdeec338fd4995c98f7170c05bcb0e5f6bfb2e1116e6f645fff835f946e",
        1_024_044,
    ),
    (
        "noise/free-sound/noise-free-sound-0298.wav",
        "2ddedf44f80d7e551533f81f6b90f6ac6850b6cfe3438ee806f5dbe5d7e3ba0a",
        1_024_044,
    ),
    (
        "noise/free-sound/noise-free-sound-0251.wav",
        "d9469909efe455ff2b094ae4b827bf812b46984a4a27dc44847eab2ef487d455",
        1_023_208,
    ),
    (
        "noise/free-sound/noise-free-sound-0733.wav",
        "89d44ffe50b58f759bfcc7ffa77ccb574c572811a73461f055638caa8754da61",
        1_022_372,
    ),
    (
        "noise/free-sound/noise-free-sound-0035.wav",
        "8ce827cfb336a35764f5b143762cd68623617d121bb3dbd1ecdeb3ce11331fb6",
        1_021_910,
    ),
    (
        "noise/free-sound/noise-free-sound-0709.wav",
        "9792ef62be558c2efff564b3587533f3ef5c9596cf58fd14a68e505a8377102c",
        1_019_492,
    ),
    // 5 longest sound-bible noise clips (≈2.4 min)
    (
        "noise/sound-bible/noise-sound-bible-0075.wav",
        "35590454624591e796ef0ec6a1ae3363ededb2a4d257f93b0ac06215b80785a9",
        1_045_772,
    ),
    (
        "noise/sound-bible/noise-sound-bible-0044.wav",
        "2f5747dfb272339422c818b32bae67ce28a2db6c93f98e1dff502f1c03e31442",
        961_350,
    ),
    (
        "noise/sound-bible/noise-sound-bible-0067.wav",
        "1191363426fc0621b0a68ebf3f10de2b0851ba189bffcff41ce3c1786ad55e7a",
        887_790,
    ),
    (
        "noise/sound-bible/noise-sound-bible-0014.wav",
        "7ec2526e4522b6f4588302f946f0dec9f9010b732c64487f8631982d1d104926",
        880_236,
    ),
    (
        "noise/sound-bible/noise-sound-bible-0032.wav",
        "aee08e2c4ca0f417dbb2dc1c57b0553a4e191eb547aa4c03417c8d10d3397ea7",
        804_908,
    ),
];

/// Inter-file silence in the continuous stream: 2 s at 16 kHz, so
/// the natural segment-boundary reset fires between files.
pub(crate) const FILE_GAP_SAMPLES: usize = 32_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_false_reaction_set_is_one_hundred_and_twenty_one() {
        assert_eq!(EXPECTED_NON_PHRASE_TOTAL, 121);
    }

    #[test]
    fn the_phrase_lists_match_their_bands() {
        assert_eq!(CONFUSABLE_PHRASES.len(), 28);
        assert_eq!(UNRELATED_PHRASES.len(), 20);
        assert_eq!(OWNER_NEGATIVE_PHRASES.len(), 10);
        assert_eq!(PHRASE_IN_PASSING_PHRASES.len(), 8);
        assert_eq!(COMMAND_PLUS_SPEECH_PHRASES.len(), 10);
        assert_eq!(
            NOISE_PROFILES.len() + NOISE_PROFILES_DETECTION_ONLY.len(),
            14
        );
    }

    #[test]
    fn the_noise_seed_bands_do_not_overlap() {
        let detection = DETECTION_NOISE_SEED_BASE..DETECTION_NOISE_SEED_BASE + 14;
        let calibration =
            CALIBRATION_NOISE_SEED_BASE..CALIBRATION_NOISE_SEED_BASE + NOISE_PROFILES.len() as u64;
        assert!(
            detection.end <= calibration.start,
            "{detection:?} {calibration:?}"
        );
    }

    #[test]
    fn the_clip_seed_bands_are_disjoint() {
        // The scalar seed ranges each set draws from. The two detection
        // confusable bands are folded into one range: band 800 and band 810
        // produce distinct (phrase, seed) pairs even where their seed numbers
        // coincide, and neither reaches the calibration confusable range.
        let mut bands = [
            (ENROLLMENT_SEED_BASE, ENROLLMENT_SEED_BASE + 10),
            (CONFUSABLE_BANDS[0].0, CONFUSABLE_BANDS[1].0 + 28),
            (UNRELATED_BANDS[0].0, UNRELATED_BANDS[1].0 + 20),
            (CONFUSABLE_SEED_BASE, CONFUSABLE_SEED_BASE + 28 * 5),
            (UNRELATED_SEED_BASE, UNRELATED_SEED_BASE + 20 * 3),
            (HELD_OUT_SEED_BASE, HELD_OUT_SEED_BASE + 40),
            (
                COMMAND_PLUS_SPEECH_SEED_BASE,
                COMMAND_PLUS_SPEECH_SEED_BASE + 10,
            ),
            (PHRASE_IN_PASSING_SEED_BASE, PHRASE_IN_PASSING_SEED_BASE + 8),
            (OWNER_NEGATIVE_SEED_BASE, OWNER_NEGATIVE_SEED_BASE + 30),
            (
                audio::CONDITIONING_NOISE_SEED_BASE,
                audio::CONDITIONING_NOISE_SEED_BASE + NUM_ENROLLMENT_VARIANTS as u64,
            ),
        ];
        bands.sort_by_key(|band| band.0);
        for pair in bands.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "seed bands {:?} and {:?} overlap",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn the_real_audio_subset_is_the_pinned_thirty_three() {
        assert_eq!(REAL_AUDIO_FILES.len(), 33);
        assert_eq!(
            REAL_AUDIO_FILES
                .iter()
                .filter(|(path, _, _)| path.starts_with("speech/"))
                .count(),
            18
        );
        assert_eq!(
            REAL_AUDIO_FILES
                .iter()
                .filter(|(path, _, _)| path.starts_with("noise/"))
                .count(),
            15
        );
        assert!(
            REAL_AUDIO_FILES
                .iter()
                .all(|(_, sha256, bytes)| sha256.len() == 64 && *bytes > 0)
        );
    }
}
