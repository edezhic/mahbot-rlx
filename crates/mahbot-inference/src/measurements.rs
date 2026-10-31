// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The repository's record: three files at its root, one per measuring surface,
//! each rewritten whole by the only run that can produce it — `STAND.md` by a
//! complete main stand run, `TRANSCRIPTION-CORPUS.md` by a complete corpus run,
//! `WAKE-WORD-BENCH.md` by a complete wake-word bench run.
//!
//! A run writes no other mode's file, and a partial, subset or screened run writes
//! nothing at all. The files are a record, not a log: history is what version
//! control is for.
//!
//! A mode takes its file before it measures — sweeping the temporaries no live run
//! is writing beside it, and proving the file writable — and writes it once, whole,
//! at the end.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A record file, and the one mode that writes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Record {
    Stand,
    TranscriptionCorpus,
    WakeWordBench,
}

impl Record {
    /// Every record, so a path that must speak of all of them — the digest's
    /// exclusions, a test — cannot be written against three names and miss a fourth.
    pub const ALL: [Self; 3] = [Self::Stand, Self::TranscriptionCorpus, Self::WakeWordBench];

    /// The file's name at the workspace root.
    #[must_use]
    pub const fn file(self) -> &'static str {
        match self {
            Self::Stand => "STAND.md",
            Self::TranscriptionCorpus => "TRANSCRIPTION-CORPUS.md",
            Self::WakeWordBench => "WAKE-WORD-BENCH.md",
        }
    }

    /// The file's heading.
    #[must_use]
    pub const fn heading(self) -> &'static str {
        match self {
            Self::Stand => "# Stand",
            Self::TranscriptionCorpus => "# Transcription corpus",
            Self::WakeWordBench => "# Wake-word bench",
        }
    }

    /// The directory the record goes into: `MAHBOT_MEASUREMENTS` when it names a
    /// directory, otherwise the workspace root of the tree that **compiled** this
    /// binary (`CARGO_MANIFEST_DIR`'s grandparent). A redirect that is not a
    /// directory would collapse the three modes into one file, so it is refused —
    /// before a run measures anything, through [`Record::prepare`].
    fn dir() -> Result<PathBuf> {
        let dir = match std::env::var_os("MAHBOT_MEASUREMENTS").filter(|path| !path.is_empty()) {
            Some(path) => PathBuf::from(path),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."),
        };
        if !dir.is_dir() {
            anyhow::bail!(
                "`{}` is not a directory — each record is written under its own name",
                dir.display()
            );
        }
        Ok(dir)
    }

    /// Where a run writes it, under the directory [`Record::dir`] resolves.
    fn path(self) -> Result<PathBuf> {
        Ok(Self::dir()?.join(self.file()))
    }

    /// The recipe this file's mode is run with — never the process's own argv, which
    /// for a bench harness is a hashed path in a build directory and for any mode
    /// carries its scratch flags. The input bullets beside it name the material the
    /// run read, which is the rest of what a re-run needs.
    const fn command(self) -> &'static str {
        match self {
            Self::Stand => "cargo run --release -p mahbot-inference",
            Self::TranscriptionCorpus => {
                "cargo run --release -p mahbot-inference --features bench --bin \
                 transcription-corpus"
            }
            Self::WakeWordBench => {
                "cargo bench -p mahbot-inference --features bench --bench wake_word -- --report <file>"
            }
        }
    }

    /// The provenance line stating the recipe this record is reproduced with.
    fn command_line(self) -> String {
        format!("- command: {}\n", self.command())
    }

    /// Take the record a run is about to write: sweep the temporaries no live run is
    /// writing beside it, then prove it writable — at the start of a run, before it
    /// measures anything, and never from inside a write. A write is where
    /// another run's in-flight temporary lives, and the per-clip cache this crate
    /// writes goes through the same writer; the sweep must never reach it.
    ///
    /// The probe carries the name `temporary` gives this writer's own temporary, so a
    /// run killed in this window leaves a file the next run of this record sweeps and
    /// `tree_digest` already leaves out.
    ///
    /// # Errors
    /// The redirect is not a directory, or nothing can be written into it.
    pub fn prepare(self) -> Result<()> {
        let path = self.path()?;
        sweep_beside(&path);
        let probe = temporary(&path, std::process::id());
        std::fs::write(&probe, b"")
            .and_then(|()| std::fs::remove_file(&probe))
            .with_context(|| format!("cannot write {}", path.display()))
    }
}

/// Replace `record` whole: its heading, its body — the table or tables, each with
/// a short caption under it — and the provenance block its inputs build. A mode
/// owns one file, so this is the only write path.
pub fn write(record: Record, body: &str, inputs: &str) -> Result<()> {
    write_at(&record.path()?, record, body, &provenance(record, inputs))
}

/// Replace a record file whole. Its previous content is irrelevant: a record is
/// the run that wrote it, and no run edits another's.
fn write_at(path: &Path, record: Record, body: &str, provenance: &str) -> Result<()> {
    let body = format!(
        "{}\n\n{}\n\n{}\n",
        record.heading(),
        body.trim(),
        provenance.trim(),
    );
    write_atomic(path, |tmp| Ok(std::fs::write(tmp, body.as_bytes())?))
}

/// The mark between a file and the pid of the temporary beside it: `<target>.tmp-<pid>`,
/// the one naming rule. `temporary` writes a name from it and `leftover` reads
/// one back, so a change to the rule cannot reach one and miss the other.
const TEMP_MARK: &str = ".tmp-";

/// The temporary [`write_atomic`] writes beside `path` for the writer `pid`:
/// `<target>.tmp-<pid>`, beside the file it is to become.
fn temporary(path: &Path, pid: u32) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(TEMP_MARK);
    name.push(pid.to_string());
    path.with_file_name(name)
}

/// The target and pid a name carries when it is a writer's temporary, and `None`
/// for any other name — a checkout's own `foo.tmp` is content like any other. The one
/// rule: the writer builds its name from it, the sweep matches a target against it,
/// and the tree digest asks whether any target matches.
fn leftover(name: &str) -> Option<(&str, u32)> {
    let (target, pid) = name.rsplit_once(TEMP_MARK)?;
    let whole =
        !target.is_empty() && !pid.is_empty() && pid.bytes().all(|byte| byte.is_ascii_digit());
    whole
        .then(|| pid.parse().ok())
        .flatten()
        .map(|pid| (target, pid))
}

/// The crate's one atomic whole-file writer: the bytes go to a temporary file beside
/// the target, that file is flushed to the device, and only then is it renamed into
/// place — so a name never becomes visible before its bytes are. The rename's
/// directory entry is flushed too, where the platform lets a directory be synced.
/// A failed write leaves the target untouched and its own temporary file removed.
/// A killed writer's temporary is reclaimed at the start of the next run that owns the
/// target — [`Record::prepare`] for a record, `bench::prepare_cache` for the corpus
/// cache, the bench's report swept beside its own path — never from here: a write is
/// where another run's in-flight temporary lives.
pub(crate) fn write_atomic(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let tmp = temporary(path, std::process::id());
    let written = write(&tmp)
        .and_then(|()| {
            Ok(std::fs::OpenOptions::new()
                .write(true)
                .open(&tmp)?
                .sync_all()?)
        })
        .and_then(|()| {
            std::fs::rename(&tmp, path)
                .with_context(|| format!("cannot move {} into place", tmp.display()))
        })
        .map(|()| {
            // Where the platform refuses to sync a directory, the file's own bytes
            // are still flushed, which is what a reader of the new name sees.
            if let Some(dir) = path.parent() {
                let _ = std::fs::File::open(dir).and_then(|dir| dir.sync_all());
            }
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written.with_context(|| format!("cannot write {}", path.display()))
}

/// Remove the temporaries a killed writer left in `dir` — `target`'s alone when it
/// names one file, any target's otherwise — and never one a live run is still
/// writing: two runs can share a directory under different locks, and deleting the
/// temporary under a live run's rename loses that run's work. A run's own pid is
/// swept too, because the run that prepares a record has written nothing yet. Where
/// the platform cannot say whether a pid is running, the leftover is left alone.
pub(crate) fn sweep(dir: &Path, target: Option<&str>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let file = entry.file_name();
        let Some((owner, pid)) = file.to_str().and_then(leftover) else {
            continue;
        };
        let named = target.is_none_or(|target| owner == target);
        if named && (pid == std::process::id() || !running(pid)) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// [`sweep`] for the file `path` is to become: the temporaries of that one name, left
/// beside it.
pub(crate) fn sweep_beside(path: &Path) {
    if let (Some(dir), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str())) {
        sweep(dir, Some(name));
    }
}

/// Whether `pid` is a running process; `true` where the platform has no reading, so
/// a leftover whose owner may be a run in flight is never removed. A killed writer's
/// temporary is therefore reclaimed on unix targets only — elsewhere it stays where it
/// lies, which the tree digest leaves out either way.
fn running(pid: u32) -> bool {
    // A writer is never pid 0, so a name that carries it is a leftover by definition
    // (and signal 0 would otherwise mean this process's own group).
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // `EPERM`: the pid exists but belongs to another user, so it is a live run.
        const EPERM: i32 = 1;

        unsafe extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        let Ok(pid) = i32::try_from(pid) else {
            return true;
        };
        // SAFETY: signal 0 sends nothing; it asks the kernel whether this pid exists.
        unsafe {
            kill(pid, 0) == 0 || std::io::Error::last_os_error().raw_os_error() == Some(EPERM)
        }
    }
    #[cfg(not(unix))]
    {
        true
    }
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
fn now() -> String {
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
fn machine() -> String {
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
fn device() -> String {
    rlx_wgpu::device::wgpu_device().map_or_else(
        || "no wgpu adapter".to_string(),
        |device| format!("{} ({:?})", device.name, device.backend),
    )
}

/// `release` or `debug`.
#[must_use]
fn profile() -> String {
    if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    }
}

/// The one provenance block every record carries: when and where the row was taken,
/// from what code, under which switches, by which invocation, and the run's own
/// input pins. `inputs` is `record`'s own bullets — the material the rows read.
#[must_use]
fn provenance(record: Record, inputs: &str) -> String {
    let mut body = format!("_Measured {}._\n", now());
    body.push_str(&format!("- machine: {}\n", machine()));
    body.push_str(&format!("- device: {}\n", device()));
    body.push_str(&format!("- profile: {}\n", profile()));
    body.push_str(&format!(
        "- code: {} (the commit and the SHA-256 of the tree's difference from it, {})\n",
        code_state(),
        exclusions_note(),
    ));
    body.push_str(&record.command_line());
    body.push_str(&run_form());
    body.push_str(inputs);
    body
}

/// The `RLX_*` and `WGPU_*` variables this run had set, so a record taken under a
/// scheduling, precision or instrumentation switch is distinguishable from a clean
/// one — some of them move the numbers. Omitted when none is set. Read as the
/// platform hands the names over, so a variable that is not UTF-8 cannot take the
/// record with it.
fn run_form() -> String {
    let mut set: Vec<(String, String)> = std::env::vars_os()
        .filter(|(name, _)| {
            let name = name.to_string_lossy();
            name.starts_with("RLX_") || name.starts_with("WGPU_")
        })
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect();
    if set.is_empty() {
        return String::new();
    }
    set.sort();
    let pairs: Vec<String> = set
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    format!("- run form: {}\n", pairs.join(" · "))
}

/// The code state the numbers were taken at: `git describe --always` plus the
/// SHA-256 of what the tree holds beyond its commit, so a reader can reconstruct
/// the content that produced a figure from the two. The digest is appended
/// whenever git can answer, and a clean tree's is the stable empty-diff constant;
/// when git cannot, the line carries the describe state alone rather than
/// inventing a figure. `git describe --dirty` counts the vendored MLX submodule,
/// which `change_digest` excludes, so the marker and the digest would otherwise
/// disagree about what "dirty" means.
fn code_state() -> String {
    let describe = std::process::Command::new("git")
        .args(["describe", "--always"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "no commit".to_string());
    match change_digest() {
        Some(digest) => format!("{describe} change-sha256 {digest}"),
        None => describe,
    }
}

/// The paths [`change_digest`] leaves out: the three records — each mode's own file is
/// the figure, not part of the code that produced it — and the vendored MLX checkout,
/// which is not this fork's content. One list, derived from [`Record::ALL`] and
/// printed beside the figure it produces.
fn exclusions() -> Vec<&'static str> {
    Record::ALL
        .into_iter()
        .map(Record::file)
        .chain(["crates/backends/rlx-mlx-sys"])
        .collect()
}

/// [`exclusions`] as the provenance line's parenthetical.
fn exclusions_note() -> String {
    let names: Vec<String> = exclusions()
        .iter()
        .map(|path| format!("`{path}`"))
        .collect();
    format!("{} left out", names.join(", "))
}

/// SHA-256 over what the tree holds beyond its commit: `git diff HEAD --binary`
/// with [`exclusions`] left out — the digest names the *code* that produced a
/// figure, and the record is the figure — plus every untracked, non-ignored
/// file's path and bytes, since an added source file the index has not seen is
/// part of that code too. A writer's leftover temporary is not content: it is left
/// out by the rule that recognizes it (`leftover`). First 16 hex characters.
/// `None` when git cannot answer; the caller then records the describe state alone
/// rather than inventing a figure.
fn change_digest() -> Option<String> {
    tree_digest(&exclusions())
}

/// [`change_digest`] over an explicit exclusion set — so the test can recompute the
/// figure from the rule the record prints, through this same plumbing.
fn tree_digest(exclude: &[&str]) -> Option<String> {
    use sha2::{Digest as _, Sha256};

    // From the repository's root, not this crate's directory: a `Path`-relative
    // pathspec and `ls-files`' own output would otherwise both stop at the crate, and
    // the digest would cover this crate's difference only.
    let root = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    let root = PathBuf::from(String::from_utf8(root.stdout).ok()?.trim().to_string());
    let args: Vec<String> = ["diff", "HEAD", "--binary", "--", "."]
        .iter()
        .map(|arg| (*arg).to_string())
        .chain(exclude.iter().map(|path| format!(":(exclude){path}")))
        .collect();
    let diff = std::process::Command::new("git")
        .args(&args)
        .current_dir(&root)
        .output()
        .ok()?;
    if !diff.status.success() {
        return None;
    }
    let untracked = std::process::Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(&root)
        .output()
        .ok()?;
    if !untracked.status.success() {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(&diff.stdout);
    let untracked = untracked_paths(&untracked.stdout)?
        .into_iter()
        .filter(|path| !exclude.contains(&path.as_str()))
        .filter(|path| {
            !Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(leftover)
                .is_some()
        })
        .collect();
    hash_untracked(&mut hasher, &root, untracked);
    Some(hex(&hasher.finalize())[..16].to_string())
}

/// `git ls-files --others --exclude-standard` output as a path list.
fn untracked_paths(bytes: &[u8]) -> Option<Vec<String>> {
    Some(
        std::str::from_utf8(bytes)
            .ok()?
            .lines()
            .filter(|path| !path.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// Feed each untracked path and its bytes into `hasher`, sorted so the digest
/// does not depend on the order git lists them. A file that cannot be read
/// (permissions, a directory the listing caught) contributes its path and a
/// marker rather than dropping the whole digest to `None`: the digest exists to
/// identify content, and answering "no change" for a tree it merely could not
/// read is the one answer it must not give. Streamed, like [`sha256_file`], so an
/// untracked artifact of any size costs a fixed buffer rather than its own length
/// in memory.
fn hash_untracked(hasher: &mut sha2::Sha256, root: &Path, mut paths: Vec<String>) {
    use sha2::Digest as _;

    paths.sort_unstable();
    for path in paths {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        match std::fs::File::open(root.join(&path)) {
            Ok(mut file) => {
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    match std::io::Read::read(&mut file, &mut buf) {
                        Ok(0) => break,
                        Ok(n) => hasher.update(&buf[..n]),
                        Err(_) => {
                            hasher.update(b"<unreadable>");
                            break;
                        }
                    }
                }
            }
            Err(_) => hasher.update(b"<unreadable>"),
        }
    }
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

    /// A directory of this test's own — so a sweep that names no target cannot reach a
    /// file another program left in the shared temp directory — removed whole when it
    /// goes out of scope, including on an assertion that fails and returns early.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let dir = std::env::temp_dir().join(format!(
                "mahbot-record-{tag}-{}-{nanos}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("a temp directory");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A process the sweep must see as a live run, ended and reaped when it goes out of
    /// scope so a failed assertion does not leave it running.
    #[cfg(unix)]
    struct Live(std::process::Child);

    #[cfg(unix)]
    impl Drop for Live {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// Each checked-in record opens with the heading its own mode writes, so a
    /// hand-edited file cannot contradict the code that owns it — and the three
    /// files' existence is the acceptance, so a missing one fails here rather than
    /// passing quietly.
    #[test]
    fn every_record_opens_with_its_own_heading() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        for record in Record::ALL {
            let path = root.join(record.file());
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert!(
                text.starts_with(record.heading()),
                "{} does not open with `{}`",
                path.display(),
                record.heading()
            );
        }
    }

    /// The provenance line names the content, not only the commit, and the rule it
    /// prints is the rule it computes: the digest must answer here (a tree git
    /// cannot read has nothing to pin, so the run fails rather than passing
    /// quietly), and a reader who applies the exclusions the printed rule names —
    /// read out of the rule itself, not out of `exclusions` — must arrive at the
    /// figure the line carries.
    /// Device-free, and it reads the tree instead of writing into it.
    #[test]
    fn the_provenance_line_carries_the_digest_of_the_trees_difference() {
        let state = code_state();
        eprintln!("code_state: {state}");
        let (_describe, digest) = state
            .split_once(" change-sha256 ")
            .unwrap_or_else(|| panic!("no digest to pin (git could not answer): {state}"));
        assert_eq!(digest.len(), 16, "16 hex characters: {state}");
        assert!(
            digest.chars().all(|c| c.is_ascii_hexdigit()),
            "hexadecimal: {state}"
        );

        let rule = exclusions_note();
        let named: Vec<&str> = rule.split('`').skip(1).step_by(2).collect();
        assert!(!named.is_empty(), "the rule names its exclusions: {rule}");
        assert_eq!(
            digest,
            tree_digest(&named).expect("git must answer"),
            "the rule the record prints must reproduce the digest it sits beside: {rule}"
        );
    }

    /// A write replaces the file whole: there are no sections, so a later write
    /// cannot leave an earlier run's body behind.
    #[test]
    fn a_write_replaces_the_whole_file() {
        let dir = Scratch::new("replace");
        let path = dir.path().join("STAND.md");
        write_at(
            &path,
            Record::Stand,
            "| a |\n|---|\n\ncaption one",
            "- code: one",
        )
        .expect("first write");
        write_at(
            &path,
            Record::Stand,
            "| b |\n|---|\n\ncaption two",
            "- code: two",
        )
        .expect("second write");
        let text = std::fs::read_to_string(&path).expect("the file");
        assert!(
            text.starts_with("# Stand\n\n| b |\n|---|\n\ncaption two"),
            "{text}"
        );
        assert!(
            !text.contains("caption one") && !text.contains("| a |"),
            "{text}"
        );
    }

    /// Each record states the recipe its own mode is reproduced with, in the line the
    /// provenance block renders from the record alone — a shell invocation, never a
    /// path from the run host, which for a bench harness is a hashed path in a build
    /// directory and for any mode carries the run's scratch flags.
    #[test]
    fn every_record_names_the_invocation_its_mode_documents() {
        for record in Record::ALL {
            let line = record.command_line();
            assert!(
                line.starts_with("- command: ") && !line.contains('/'),
                "{}: {line}",
                record.file()
            );
        }
    }

    /// The naming rule exists once: the name the writer makes for a pid is the name the
    /// sweep and the tree digest recognize back, and nothing else is a temporary.
    #[test]
    fn the_temporary_rule_is_one_rule() {
        let made = temporary(Path::new("STAND.md"), 12);
        let name = made
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a name");
        assert_eq!(name, "STAND.md.tmp-12");
        assert_eq!(leftover(name), Some(("STAND.md", 12)));
        assert_eq!(leftover("a.f32.tmp-0"), Some(("a.f32", 0)));
        for other in ["STAND.md.tmp-notes", "STAND.md.tmp-", ".tmp-12", "STAND.md"] {
            assert_eq!(leftover(other), None, "`{other}` is not a temporary");
        }
    }

    /// A write touches nothing but its own temporary: the per-clip cache goes through
    /// this same writer, ~2 000 times per corpus build, and a second run can share
    /// that directory under another lock while its clips are in flight.
    #[test]
    fn a_write_never_removes_a_file_it_did_not_make() {
        let dir = Scratch::new("write-isolation");
        let path = dir.path().join("STAND.md");
        let others = [temporary(&path, 0), temporary(&path, 999_999)];
        for file in &others {
            std::fs::write(file, b"another run's clip").expect("a file beside the target");
        }

        write_at(&path, Record::Stand, "| a |", "- code: one").expect("the write");

        for file in &others {
            assert!(file.exists(), "{} was removed by a write", file.display());
        }
    }

    /// A run killed between the write and the rename leaves `<record>.tmp-<pid>`
    /// beside it, and staging would pick it up — the write path a killed run never
    /// reaches. The sweep runs at the start of the next run, so a subset, screened or
    /// failed run clears it too, and it clears that record's name only: a directory can
    /// hold another writer's files. A file that merely looks temporary is content and
    /// stays, and so does the temporary a live run is still writing — a second run can
    /// share a directory under another lock. A directory sweep with no target (the
    /// corpus cache) reaps every dead writer's leftover, and no live one's.
    #[cfg(unix)]
    #[test]
    fn a_sweep_reaps_only_what_no_live_run_is_writing() {
        let dir = Scratch::new("sweep");
        let path = dir.path().join("STAND.md");
        write_at(&path, Record::Stand, "| a |", "- code: one").expect("a first write");
        // This process's own (a reused pid's leftover), a pid no writer can have, a name
        // that only looks temporary, a live run's in-flight write, and another file's
        // killed temporary.
        let own = temporary(&path, std::process::id());
        let stale = temporary(&path, 999_999);
        let foreign = dir.path().join(format!("STAND.md{TEMP_MARK}notes"));
        let sleeper = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("a live process");
        let live = temporary(&path, sleeper.id());
        let _live = Live(sleeper);
        let neighbour = temporary(&dir.path().join("neighbour.f32"), 999_999);
        for file in [&own, &stale, &foreign, &live, &neighbour] {
            std::fs::write(file, b"half a write").expect("a file beside the record");
        }

        sweep_beside(&path);

        assert!(!stale.exists(), "{} survived the sweep", stale.display());
        assert!(!own.exists(), "{} survived the sweep", own.display());
        assert!(foreign.exists(), "{} was swept", foreign.display());
        assert!(
            live.exists(),
            "{} is another run's in-flight write",
            live.display()
        );
        assert!(
            neighbour.exists(),
            "{} is another writer's file",
            neighbour.display()
        );

        sweep(dir.path(), None);

        assert!(!neighbour.exists(), "a dead writer's neighbour survived");
        assert!(live.exists(), "a live run's in-flight write was swept");
    }
}
