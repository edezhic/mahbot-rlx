// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one location every input lives at.
//!
//! There is no configuration: the crate resolves its model files, the pinned
//! corpus and the music pool at one fixed path per operating system, and nowhere
//! else. Nothing here is created by the crate — the tree is provisioned by hand
//! and its absence is an error naming the path — except `wake-corpus/`, the
//! synthesized corpus the wake-word build owns ([`crate::bench`]).
//!
//! The layout inside the root:
//!
//! * the five model files, under the names they are published under;
//! * `corpus/real-audio/` — the pinned real-audio corpus, in its own structure;
//! * `corpus/music/` — the music pool, with the list its code treats as
//!   authoritative;
//! * `wake-corpus/` — the synthesized corpus the wake-word build renders: one
//!   clip per phrase and voice, and the clone spans cut from the corpus.

use std::path::PathBuf;

use anyhow::{Result, bail};

/// The root: macOS.
#[cfg(target_os = "macos")]
pub const ROOT: &str = "/Users/Shared/MahBot/models";
/// The root: Linux.
#[cfg(target_os = "linux")]
pub const ROOT: &str = "/var/lib/mahbot/models";
/// The root: Windows.
#[cfg(target_os = "windows")]
pub const ROOT: &str = r"C:\ProgramData\MahBot\models";
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
compile_error!(
    "mahbot-inference has one input location per operating system: macOS, Linux and Windows"
);

/// The root as a path.
#[must_use]
pub fn root() -> PathBuf {
    PathBuf::from(ROOT)
}

/// The pinned real-audio corpus.
#[must_use]
pub fn corpus() -> PathBuf {
    root().join("corpus").join("real-audio")
}

/// The music pool.
#[must_use]
pub fn music() -> PathBuf {
    root().join("corpus").join("music")
}

/// The synthesized wake corpus — the one directory the crate may create, and only
/// inside an existing root.
#[must_use]
pub fn wake_corpus() -> PathBuf {
    root().join("wake-corpus")
}

/// `dir`, or the error naming it: the tree is provisioned by hand, so an absent
/// directory is a failure and never a created one or a substitute.
pub(crate) fn require_dir(dir: PathBuf, what: &str) -> Result<PathBuf> {
    if !dir.is_dir() {
        bail!("the {what} directory {} does not exist", dir.display());
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The root and the subtree names are the installed tree's address — casing
    /// included (`MahBot` on macOS and Windows, `mahbot` on Linux) — so editing one
    /// here is a change of the machine's layout, not a refactor.
    #[test]
    fn the_layout_is_the_installed_one() {
        #[cfg(target_os = "macos")]
        assert_eq!(ROOT, "/Users/Shared/MahBot/models");
        #[cfg(target_os = "linux")]
        assert_eq!(ROOT, "/var/lib/mahbot/models");
        #[cfg(target_os = "windows")]
        assert_eq!(ROOT, r"C:\ProgramData\MahBot\models");
        let suffix = |path: PathBuf| {
            path.strip_prefix(root())
                .expect("below the root")
                .to_path_buf()
        };
        assert_eq!(suffix(corpus()), Path::new("corpus/real-audio"));
        assert_eq!(suffix(music()), Path::new("corpus/music"));
        assert_eq!(suffix(wake_corpus()), Path::new("wake-corpus"));
    }

    /// An absent directory is an error that names the full path — the contract
    /// every resolver in the crate is built on.
    #[test]
    fn an_absent_directory_is_named() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let absent = std::env::temp_dir().join(format!(
            "mahbot-inference-absent-input-{}-{nanos}",
            std::process::id()
        ));
        let error = format!("{:#}", require_dir(absent.clone(), "model").unwrap_err());
        assert!(error.contains(&absent.display().to_string()), "{error}");
    }
}
