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

use crate::source::rust_code;

/// A record file, and the one mode that writes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Record {
    Stand,
    TranscriptionCorpus,
    WakeWordBench,
}

impl Record {
    /// Every record, so a test that must speak of all of them cannot be written
    /// against three names and miss a fourth.
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

/// Replace `record` whole: its heading, its body — the table or tables — and the
/// provenance block its inputs build. A mode owns one file, so this is the only
/// write path.
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

/// The machine's load average when the record is written, as the kernel states it, or
/// empty where it cannot be read. A figure taken under another process's load is not
/// comparable with one taken on a quiet machine, and two records cannot be subtracted
/// without knowing which is which — so the take states it.
#[must_use]
fn load() -> String {
    let raw = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        // Linux's procps `sysctl` fails with empty stdout; without this the empty
        // success short-circuits the fallback that exists for exactly that platform.
        .filter(|text| !text.is_empty())
        .or_else(|| std::fs::read_to_string("/proc/loadavg").ok());
    // Both forms lead with the three figures; macOS wraps them in `{ }` and Linux
    // appends a `running/total` pair, neither of which is one.
    let figures: Vec<&str> = raw
        .as_deref()
        .unwrap_or_default()
        .split_whitespace()
        .filter(|word| word.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .take(3)
        .collect();
    figures.join(" · ")
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
    let load = load();
    if !load.is_empty() {
        body.push_str(&format!("- load: {load} (1m · 5m · 15m)\n"));
    }
    body.push_str(&format!(
        "- code: {} ({})\n",
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
/// SHA-256 of the tree's content, so a reader can identify the code that produced a
/// figure. The commit is only a pointer to where the numbers came from; the digest
/// is the identity, and it is taken over content rather than over a difference from
/// the commit, so two different clean trees do not share it. When git cannot answer
/// or a file cannot be read, the line carries `tree-sha256 unavailable` rather than
/// inventing a figure or silently omitting the token.
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
    match tree_digest(exclusions()) {
        Some(digest) => format!("{describe} tree-sha256 {digest}"),
        // A digest that could not be taken is named, never omitted: a missing token
        // would read as a record written before the rule existed.
        None => format!("{describe} tree-sha256 unavailable"),
    }
}

/// What the content digest leaves out, and why: a document is prose about the tree
/// rather than an input to a figure (the three records are Markdown too), no file
/// under a `tests` directory or in a `tests.rs` module runs in a measuring binary,
/// and the vendored MLX checkout is not this fork's content. One list, printed beside
/// the figure it produces; [`excluded`] matches it.
fn exclusions() -> &'static [&'static str] {
    &["*.md", "tests", "tests.rs", "crates/backends/rlx-mlx-sys"]
}

/// [`exclusions`] as the provenance line's parenthetical: the rule the digest is
/// computed by, including the transforms that cannot be read out of a path list.
fn exclusions_note() -> String {
    let names: Vec<String> = exclusions()
        .iter()
        .map(|path| format!("`{path}`"))
        .collect();
    format!(
        "SHA-256 of the tree's content, not a difference from the commit: every file \
         git tracks or lists as untracked and not ignored is read (ignored build \
         output is not; a tracked file deleted from the worktree is marked absent); in \
         Rust sources comments are stripped (literals respected), `#[cfg(test)]` and \
         `#[cfg(all(test, ...))]` items are cut, and whitespace outside literals is \
         collapsed; {} and writer temporaries are left out; an unreadable file leaves \
         the digest unavailable",
        names.join(", ")
    )
}

/// Whether `path` is left out by one of `exclude`'s rules: an exact path or directory
/// prefix, a `*.md` suffix, a `tests` path component, or a `tests.rs` name.
fn excluded(path: &str, exclude: &[&str]) -> bool {
    if Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(leftover)
        .is_some()
    {
        return true;
    }
    exclude.iter().any(|rule| match *rule {
        "*.md" => path.ends_with(".md"),
        "tests" => path.split('/').any(|component| component == "tests"),
        "tests.rs" => path == "tests.rs" || path.ends_with("/tests.rs"),
        rule => path == rule || path.starts_with(&format!("{rule}/")),
    })
}

/// SHA-256 over the tree's content, with [`exclusions`] and [`rust_code`] applied,
/// first 16 hex characters. Every tracked file and every untracked, non-ignored
/// file is read; a tracked file missing from the worktree is marked absent, so a
/// deletion moves the digest rather than reading as the complete tree. A file that
/// is there but cannot be read — a broken symlink included — fails the whole digest
/// rather than dropping out of it. `None` when git cannot list the tree or a file
/// cannot be read: the caller then names the digest unavailable, never a digest of a
/// tree it could not read.
fn tree_digest(exclude: &[&str]) -> Option<String> {
    use sha2::{Digest as _, Sha256};

    // A path is followed by its state and, for a file, its bytes: a deletion, a
    // directory (a gitlink) and an empty file are three different things.
    const ABSENT: u8 = 0;
    const FILE: u8 = 1;
    const DIRECTORY: u8 = 2;

    // From the repository's root, not this crate's directory: `ls-files` run in the
    // crate would cover this crate only.
    let root = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    let root = PathBuf::from(String::from_utf8(root.stdout).ok()?.trim().to_string());
    // NUL-separated: a path may hold any other byte. `--cached` lists the index, so
    // a worktree deletion is still listed; the metadata check below marks it absent.
    let listed = std::process::Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(&root)
        .output()
        .ok()?;
    if !listed.status.success() {
        return None;
    }
    let mut paths: Vec<String> = std::str::from_utf8(&listed.stdout)
        .ok()?
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .collect();
    paths.sort_unstable();
    paths.dedup();

    // Only a Rust source is held whole, because only it is lexed; every other file
    // streams through this fixed buffer, so a large untracked artifact costs a
    // constant here — the process writing a memory record must not allocate a
    // file-sized buffer to finish its own bookkeeping.
    let mut buf = vec![0u8; 64 * 1024];
    let mut hasher = Sha256::new();
    for path in paths {
        if excluded(&path, exclude) {
            continue;
        }
        hasher.update(path.as_bytes());
        hasher.update([0]);
        let full = root.join(&path);
        // `symlink_metadata` does not follow: a path that is not there at all is a
        // deletion and is marked, while a symlink that exists is a file the digest
        // must read (a broken one fails it below).
        let link = match std::fs::symlink_metadata(&full) {
            Ok(meta) => meta,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                hasher.update([ABSENT]);
                continue;
            }
            Err(_) => return None,
        };
        if link.is_dir()
            || (link.file_type().is_symlink()
                && std::fs::metadata(&full).is_ok_and(|target| target.is_dir()))
        {
            // A gitlink (submodule) or a symlink to a directory: no file bytes.
            hasher.update([DIRECTORY]);
            continue;
        }
        let mut file = match std::fs::File::open(&full) {
            Ok(file) => file,
            // The path existed a moment ago; if it cannot be opened now, the digest
            // must fail rather than drop a file it could not read.
            Err(_) => return None,
        };
        hasher.update([FILE]);
        if path.ends_with(".rs") {
            let mut bytes = Vec::new();
            if std::io::Read::read_to_end(&mut file, &mut bytes).is_err() {
                return None;
            }
            let content = rust_code(&bytes);
            hasher.update(&content);
            hasher.update(
                u64::try_from(content.len())
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
        } else {
            let mut len = 0u64;
            loop {
                let n = match std::io::Read::read(&mut file, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => return None,
                };
                len += n as u64;
                hasher.update(&buf[..n]);
            }
            hasher.update(len.to_le_bytes());
        }
    }
    Some(hex(&hasher.finalize())[..16].to_string())
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
    use crate::report::Basis;

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

    /// The fork's checkout root, when this crate is built inside it. The records
    /// themselves are the test: a packaged or vendored copy sits under some other
    /// root and carries none, so both checks below skip there instead of failing.
    fn repository_root() -> Option<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        Record::ALL
            .iter()
            .all(|record| root.join(record.file()).is_file())
            .then_some(root)
    }

    /// Each checked-in record opens with the heading its own mode writes, and nothing
    /// else on that line, so a hand-edited file cannot contradict the code that owns
    /// it. The files live in the repository, not in the crate, so a copy outside the
    /// checkout skips this.
    #[test]
    fn every_record_opens_with_its_own_heading() {
        let Some(root) = repository_root() else {
            eprintln!("skipped: no records beside this crate");
            return;
        };
        for record in Record::ALL {
            let path = root.join(record.file());
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(
                text.lines().next().map(str::trim),
                Some(record.heading()),
                "{} does not open with `{}`",
                path.display(),
                record.heading()
            );
        }
    }

    /// The `code_state` digest is taken over the tree's content, and the rule the
    /// provenance line prints names every path the digest excludes, so the rule a
    /// record states is the rule the code applies. The transforms the rule states are
    /// the stripper's, covered by the tests in `source`; hand-reproducibility of the
    /// whole digest is deliberately not claimed. A copy outside the checkout carries
    /// no records, so it skips; in the repository a tree git cannot read has nothing
    /// to pin, so the check fails rather than passing quietly. Device-free, and it
    /// reads the tree instead of writing into it.
    #[test]
    fn the_code_state_carries_the_digest_of_the_trees_content() {
        if repository_root().is_none() {
            eprintln!("skipped: no records beside this crate");
            return;
        }
        let state = code_state();
        eprintln!("code_state: {state}");
        let (_describe, digest) = state
            .split_once(" tree-sha256 ")
            .expect("git must answer inside the repository");
        assert_eq!(digest.len(), 16, "16 hex characters: {state}");
        assert!(
            digest.chars().all(|c| c.is_ascii_hexdigit()),
            "hexadecimal: {state}"
        );
        let rule = exclusions_note();
        for path in exclusions() {
            assert!(
                rule.contains(&format!("`{path}`")),
                "the printed rule names `{path}`: {rule}"
            );
        }
    }

    /// The digest leaves out the three records, every Markdown document, every
    /// `tests/` directory, every `tests.rs` module and the vendored backend, and a
    /// writer's leftover temporary; a source file under `src/` is content.
    #[test]
    fn the_digest_excludes_documents_tests_and_the_vendored_backend() {
        let rule = exclusions();
        for record in Record::ALL {
            assert!(
                excluded(record.file(), rule),
                "{} is not excluded",
                record.file()
            );
        }
        for path in [
            "README.md",
            "docs/records.md",
            "crates/mahbot-inference/tests/product_checks.rs",
            "crates/core/rlx-ir/src/rlxmodel/tests.rs",
            "crates/backends/rlx-mlx-sys/src/lib.rs",
            "STAND.md.tmp-12",
        ] {
            assert!(excluded(path, rule), "{path} is not excluded");
        }
        for path in [
            "crates/mahbot-inference/src/measurements.rs",
            "crates/backends/rlx-wgpu/src/kernels/matmul.wgsl",
            "Cargo.toml",
            "crates/mahbot-inference/benches/wake_word.rs",
        ] {
            assert!(!excluded(path, rule), "{path} is excluded");
        }
    }

    /// A record is headings, tables, status tokens and provenance fields: no caption,
    /// narrative, note or footer. The writers produce that shape; this checks the
    /// checked-in files too, so nothing can put prose back by hand.
    #[test]
    fn the_records_carry_tables_and_provenance_only() {
        let Some(root) = repository_root() else {
            eprintln!("skipped: no records beside this crate");
            return;
        };
        let rule = exclusions_note();
        for record in Record::ALL {
            let path = root.join(record.file());
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(
                prose(&text, &rule),
                None,
                "{} is not a numbers-only record",
                record.file()
            );
            let memory = text
                .split_once("_Measured ")
                .expect("prose found a provenance block")
                .1
                .lines()
                .find_map(|line| line.trim().strip_prefix("- memory: "));
            let valid = [
                Basis::Footprint.field(),
                Basis::Resident.field(),
                Basis::Mixed.field(),
            ];
            assert!(
                memory.is_some_and(|token| valid.contains(&token)),
                "{} does not name its memory basis: {memory:?}",
                record.file()
            );
        }
    }

    /// Why `text` is not a numbers-only record, or `None` when it is one. A body line
    /// is blank, a table row or a heading; a sentence terminator anywhere in a body or
    /// provenance line is prose; a `code:` parenthetical is exempt only when it is
    /// exactly `rule`, and a `run form:` value is exempt because an environment
    /// variable is arbitrary text. The terminator set is deliberately this small: a
    /// gate cannot tell a short value from a sentence.
    fn prose(text: &str, rule: &str) -> Option<String> {
        let Some((body, provenance)) = text.split_once("_Measured ") else {
            return Some("no provenance block".to_string());
        };
        for line in body.lines() {
            let line = line.trim();
            if !(line.is_empty() || line.starts_with('|') || line.starts_with('#')) {
                return Some(format!("prose: `{line}`"));
            }
            if terminator(line) {
                return Some(format!("sentence: `{line}`"));
            }
        }
        for line in format!("_Measured {provenance}").lines() {
            let line = line.trim();
            if !(line.is_empty() || line.starts_with("_Measured ") || line.starts_with("- ")) {
                return Some(format!("prose in provenance: `{line}`"));
            }
            if let Some(value) = line.strip_prefix("- code: ") {
                if let Some((head, note)) = value.split_once(" (") {
                    let note = note.strip_suffix(')').unwrap_or(note);
                    if note != rule {
                        return Some(format!(
                            "a digest rule that is not the printed one: `{line}`"
                        ));
                    }
                    if terminator(head) {
                        return Some(format!("sentence in the code field: `{line}`"));
                    }
                } else if terminator(value) {
                    return Some(format!("sentence in the code field: `{line}`"));
                }
            } else if let Some(value) = line.strip_prefix("_Measured ") {
                // The timestamp's own closing period is the italic's, not prose.
                let value = value.strip_suffix('_').unwrap_or(value);
                let value = value.strip_suffix('.').unwrap_or(value);
                if terminator(value) {
                    return Some(format!("sentence in the timestamp: `{line}`"));
                }
            } else if line.starts_with("- run form: ") {
                // The value is whatever an environment variable held during a
                // legitimate take, so it is exempt like the `code:` rule.
            } else if terminator(line) {
                return Some(format!("sentence in provenance: `{line}`"));
            }
        }
        None
    }

    /// Whether `text` carries a sentence terminator — the one shape no unit, status
    /// token or label in these files takes.
    fn terminator(text: &str) -> bool {
        text.contains(". ") || text.contains("; ")
    }

    /// The gate's jobs: a heading is allowed, a sentence in a body or provenance line
    /// is rejected, a `code:` parenthetical is accepted only when it is the printed
    /// rule, and a `run form:` value is exempt.
    #[test]
    fn the_prose_gate_accepts_a_heading_and_rejects_a_sentence_or_another_rule() {
        let rule = exclusions_note();
        let good = format!(
            "# Stand\n\n## Voices\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n_Measured now._\n\
             - code: abc tree-sha256 0123456789abcdef ({rule})\n- memory: footprint\n"
        );
        assert_eq!(prose(&good, &rule), None);

        let old_rule = "# Stand\n\n_Measured now._\n- code: abc change-sha256 0123456789abcdef \
                        (the commit and the SHA-256 of the tree's difference from it, left out)\n\
                        - memory: footprint\n";
        assert!(prose(old_rule, &rule).is_some());

        let sentence = "# Stand\n\n| a sentence. Another sentence. |\n|---|---|\n\n_Measured now._\n\
                        - memory: footprint\n";
        assert!(prose(sentence, &rule).is_some());

        // A run form is an environment variable's value, whatever it holds.
        let run_form = "# Stand\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n_Measured now._\n\
                        - run form: RLX_TRACE=a. b\n- memory: footprint\n";
        assert_eq!(prose(run_form, &rule), None);
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
            "| a |\n|---|\n\nbody one",
            "- code: one",
        )
        .expect("first write");
        write_at(
            &path,
            Record::Stand,
            "| b |\n|---|\n\nbody two",
            "- code: two",
        )
        .expect("second write");
        let text = std::fs::read_to_string(&path).expect("the file");
        assert!(
            text.starts_with("# Stand\n\n| b |\n|---|\n\nbody two"),
            "{text}"
        );
        assert!(
            !text.contains("body one") && !text.contains("| a |"),
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
