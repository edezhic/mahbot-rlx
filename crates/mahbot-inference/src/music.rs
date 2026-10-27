// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The music pool: the one measurement input the pinned real-audio corpus does
//! not hold, and the ground truth the song rule is tested against.
//!
//! It lives here rather than in `transcribe` because it is the record's input, not
//! the product's: the surface a caller transcribes with opens any audio it is
//! given, and nothing it does depends on this listing. The split is the truth —
//! `instrumental/` holds music with no words (a transcript over any of it is the
//! defect this pool exists to catch) and `vocal/` holds singing, which the song
//! rule transcribes only where the words come out.
//!
//! One table, not three: [`CLIPS`] carries each clip's path, hash, size and
//! source, so a renamed file cannot leave the record's provenance and the gates
//! describing different files.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};

/// One pinned clip: where it lives in the pool, what it hashes, and what it is the
/// ground truth for — the licence and the expectation, as the record prints them.
pub struct Clip {
    pub path: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub source: &'static str,
}

/// The pool's pins. They were copied from the pool's own `list.txt` (which stays
/// authoritative) and every SHA-256 and byte size was re-checked when they were
/// taken.
pub const CLIPS: &[Clip] = &[
    Clip {
        path: "instrumental/01-orchestra.wav",
        sha256: "e3908b8fa57d1d6f2b8fd9dd18af71a1b2e9c07c6bd38d5aad522dbd1cc7bd2d",
        bytes: 640_078,
        source: "United States Navy Band, La Bayamesa: public domain; must not be transcribed",
    },
    Clip {
        path: "instrumental/02-solo-piano.wav",
        sha256: "6d615cb99d6e746cebdd021fce3e4870d5a695c63d7310c28043d398792cd5cd",
        bytes: 800_078,
        source: "Gershwin, Rhapsody in Blue piano solo: public domain; must not be transcribed",
    },
    Clip {
        path: "instrumental/03-electronic.wav",
        sha256: "82e1d0b280ea5ea33c932352b02c55c03b6e95815444bbc6146545ab8b1273b7",
        bytes: 640_078,
        source: "Loyalty Freak Music, High Technologic Beat Explosion: CC0 1.0; must not be \
                 transcribed",
    },
    Clip {
        path: "instrumental/04-ambient.wav",
        sha256: "4d2e5a6bb829984d87075069fc08d8fb8f79fdf48699b3704aabde06a295f1f4",
        bytes: 800_078,
        source: "Wilfredor, Ambient music test (Yamaha CK61): CC0 1.0; must not be transcribed",
    },
    Clip {
        path: "vocal/01-acappella-solfege.wav",
        sha256: "bd0908f22b3b99af8255878070f09461a8e4f9a6ac0f69004a1879dfea6d960e",
        bytes: 640_078,
        source: "anonymous, Star in the east solfege: CC BY 2.5, \
                 https://creativecommons.org/licenses/by/2.5/; shape-note solfège singing, no \
                 transcript required and its absence is not a miss, the tagger should answer \
                 singing/choir/a capella rather than an instrument",
    },
    Clip {
        path: "vocal/02-pop-lyrics.wav",
        sha256: "3c7983c16ad84de1bcbc01a2ccb6dc85018dd561afcf359390ed599d28e95bcf",
        bytes: 640_078,
        source: "Irving Berlin, Billy Murray (Edison), I Want to Go Back to Michigan: public \
                 domain; clear English lyrics, expected to be transcribed",
    },
    Clip {
        path: "vocal/03-vocal-over-band.wav",
        sha256: "37bfb80d484778c9174b213ecca04441fc3668f61d7578a0321c54637bcc00ef",
        bytes: 640_078,
        source: "Norworth and Bayes, Ada Jones and Billy Murray, Shine On, Harvest Moon: public \
                 domain; voice over an orchestra, borderline — record which way it went",
    },
    Clip {
        path: "vocal/04-vocal-indian-classical.wav",
        sha256: "2ebd58c1d4e65c1302d6c12e72ac8b372b188c34bee7caf7ddf3d7bb87e659bd",
        bytes: 640_078,
        source: "Master Madan, Yun Na Reh Reh Kar Hamein Tarsaiye: public domain; vocal with \
                 tabla, not English, no text expectation — record what comes out",
    },
];

/// Where the pool lives on the machine that supplies it.
pub const DIR_ENV: &str = "MAHBOT_ASR_MUSIC";

/// The pool's directory, from [`DIR_ENV`] (no default: the checkout carries no
/// audio).
pub fn dir() -> Result<PathBuf> {
    match std::env::var_os(DIR_ENV) {
        Some(v) if !v.is_empty() => Ok(PathBuf::from(v)),
        _ => bail!("{DIR_ENV} is not set — point it at the music pool's directory"),
    }
}

/// Every pinned clip present with its size and sha256 — the same gate the models
/// get, run before any measurement starts. Returns the absolute paths in listing
/// order. A missing or mismatched file is an error naming it.
pub fn verify() -> Result<Vec<PathBuf>> {
    let dir = std::path::absolute(dir()?)?;
    let mut paths = Vec::with_capacity(CLIPS.len());
    for clip in CLIPS {
        let path = dir.join(clip.path);
        let len = std::fs::metadata(&path)
            .with_context(|| format!("{} is missing from {}", clip.path, dir.display()))?
            .len();
        if len != clip.bytes {
            bail!(
                "{} is {len} bytes, the pin says {}",
                path.display(),
                clip.bytes
            );
        }
        let actual = crate::measurements::sha256_file(&path)?;
        if actual != clip.sha256 {
            bail!(
                "{} hashes {actual}, the pin says {}",
                path.display(),
                clip.sha256
            );
        }
        paths.push(path);
    }
    Ok(paths)
}

/// The pool's provenance as one sentence, for the record: every clip's path, what
/// it is and its licence, so the CC BY 2.5 clip's attribution travels with the
/// pins.
pub fn provenance() -> String {
    CLIPS
        .iter()
        .map(|clip| format!("{} — {}.", clip.path, clip.source))
        .collect::<Vec<_>>()
        .join(" ")
}
