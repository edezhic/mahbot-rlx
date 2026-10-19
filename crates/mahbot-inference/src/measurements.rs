// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MEASUREMENTS.md` at the workspace root: the latest numbers from this
//! crate's two measuring surfaces, replaced in place by the run that produced
//! them.
//!
//! The file is the repository's own record, not a log: each surface owns one
//! section, each section carries its own date, and a run rewrites its section
//! rather than appending to it. History is what version control is for.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The file, at the workspace root of the tree that **compiled** this binary
/// (`CARGO_MANIFEST_DIR`'s grandparent), unless `MAHBOT_MEASUREMENTS` names a
/// path. The variable is how a run that must not touch the tree — a QA run
/// checking the bench end to end — redirects the record. A release binary run
/// from elsewhere otherwise still writes into that tree. Both surfaces treat a
/// refresh that fails as a failed run: the numbers a change produces are
/// incomplete until this file shows them.
#[must_use]
pub fn path() -> PathBuf {
    match std::env::var_os("MAHBOT_MEASUREMENTS").filter(|path| !path.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("MEASUREMENTS.md"),
    }
}

const HEADER: &str = "# Measurements

The latest numbers from this crate's two measuring surfaces. Each section is
replaced by the run that produced it and carries its own date; the two surfaces
are run at different times, so one half may be older than the other. This file
is not a log — history is what version control is for.

";

/// Replace one surface's section of the file at [`path`], leaving the other as
/// it was.
pub fn refresh(section: &str, body: &str) -> Result<()> {
    refresh_at(&path(), section, body)
}

/// [`refresh`] against an explicit file, so the section arithmetic is testable
/// without touching the tree the crate was compiled in.
fn refresh_at(path: &Path, section: &str, body: &str) -> Result<()> {
    let existing = match std::fs::read_to_string(path) {
        Ok(text) => text,
        // Absent is the first run of the first surface: start from the header.
        // Any other failure leaves a file whose other section must not be
        // overwritten with an empty one.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read {}", path.display()));
        }
    };
    let begin = format!("<!-- BEGIN {section} -->");
    let end = format!("<!-- END {section} -->");
    let block = format!("{begin}\n{}\n{end}", body.trim_end());
    let text = match (existing.find(&begin), existing.find(&end)) {
        (Some(a), Some(b)) if b > a => {
            format!("{}{block}{}", &existing[..a], &existing[b + end.len()..])
        }
        _ => {
            let mut text = if existing.trim().is_empty() {
                HEADER.to_string()
            } else {
                existing
            };
            if !text.ends_with('\n') {
                text.push('\n');
            }
            if !text.ends_with("\n\n") {
                text.push('\n');
            }
            format!("{text}{block}\n")
        }
    };
    let tmp = path.with_extension(format!("md.{}.tmp", std::process::id()));
    if let Err(error) = std::fs::write(&tmp, text.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("cannot write {}", tmp.display()));
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("cannot move {} into place", tmp.display()));
    }
    Ok(())
}

/// A path as the provenance block writes it: relative to `$HOME` when it is
/// under it, so the record does not carry one machine's user name.
#[must_use]
pub fn short_path(path: &Path) -> String {
    let home = std::env::var_os("HOME").filter(|home| !home.is_empty());
    match home.and_then(|home| path.strip_prefix(&home).ok().map(|rest| rest.to_path_buf())) {
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// `2026-10-18 21:20 UTC`.
#[must_use]
pub fn now() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d %H:%M UTC"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(
            || "unknown date".to_string(),
            |text| text.trim().to_string(),
        )
}

/// The machine: architecture and CPU model.
#[must_use]
pub fn machine() -> String {
    let arch = std::process::Command::new("uname")
        .arg("-m")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(
            || "unknown arch".to_string(),
            |text| text.trim().to_string(),
        );
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|text| {
                    text.lines()
                        .find(|line| line.starts_with("model name"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|value| value.trim().to_string())
                })
        })
        .unwrap_or_else(|| "unknown CPU".to_string());
    format!("{arch} · {cpu}")
}

/// The device the graphs ran on, as the backend named it.
#[must_use]
pub fn device() -> String {
    rlx_wgpu::device::wgpu_device().map_or_else(
        || "no wgpu adapter".to_string(),
        |device| format!("{} ({:?})", device.name, device.backend),
    )
}

/// `release` or `debug`.
#[must_use]
pub fn profile() -> String {
    if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    }
}

/// The code state the numbers were taken at: `git describe --always --dirty`, so
/// a tree that differs from its commit says so — the numbers are the tree's, not
/// the commit's.
pub fn code_state() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "no commit".to_string())
}

/// A file's byte size and SHA-256, for the provenance block — the same hasher
/// the input pins use, so a model file is hashed once per run, not twice.
#[must_use]
pub fn digest(path: &Path) -> Option<(u64, String)> {
    let bytes = std::fs::metadata(path).ok()?.len();
    Some((bytes, sha256_file(path).ok()?))
}

/// Hex SHA-256 of a file's bytes, streamed so a multi-gigabyte model file is not
/// held in memory. The one hasher: the input pins, the corpus pins and the
/// provenance block all go through it.
pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("cannot read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Lowercase hex.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "mahbot-measurements-{tag}-{}-{nanos}.md",
            std::process::id()
        ))
    }

    /// One surface's refresh replaces its own block and leaves the other's
    /// exactly as it was — the failure that would silently drop a stale half.
    #[test]
    fn a_refresh_replaces_its_section_and_keeps_the_other() {
        let path = temp_file("sections");
        refresh_at(&path, "stand", "stand v1").expect("first section");
        refresh_at(&path, "bench", "bench v1").expect("second section");
        let text = std::fs::read_to_string(&path).expect("the file");
        assert!(text.contains("stand v1") && text.contains("bench v1"));
        assert_eq!(text.matches("<!-- BEGIN stand -->").count(), 1);

        refresh_at(&path, "stand", "stand v2").expect("replacement");
        let text = std::fs::read_to_string(&path).expect("the file");
        assert!(text.contains("stand v2") && !text.contains("stand v1"));
        assert!(text.contains("bench v1"), "the other section survived");
        assert_eq!(text.matches("<!-- BEGIN stand -->").count(), 1);
        let _ = std::fs::remove_file(&path);
    }

    /// An unreadable file is an error, never an empty one: rewriting it would
    /// destroy the section this run does not own.
    #[test]
    fn an_unreadable_file_is_refused_rather_than_emptied() {
        let path = temp_file("unreadable");
        std::fs::create_dir_all(&path).expect("a directory where the file should be");
        assert!(refresh_at(&path, "stand", "body").is_err());
        let _ = std::fs::remove_dir(&path);
    }
}
