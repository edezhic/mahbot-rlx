// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where the bench's inputs come from — an argument first, the environment
//! second — and the provenance pin the report records for every file that fed
//! the measurement.
//!
//! Every path is an input except the synthesized corpus, which lives in one
//! fixed, durable cache directory so a run can only ever reuse it: the checkout
//! carries no models and no corpus, so nothing is resolved relative to the crate,
//! and the report is an input too rather than a file written into the model
//! directory. The single-instance lock defaults to one shared path
//! ([`crate::guard::lock_path`]) so the bench and the stand cannot run at once.
//! A required input that is missing is an error naming both its flag and its
//! environment variable, and a path that does not exist is an error naming the
//! path.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

use crate::loader::{self, Role};
use crate::measurements::{hex, sha256_file};

use super::sets;
use crate::guard::{DEFAULT_MAX_RSS_MIB, DEFAULT_TIMEOUT_SECS};

/// An input the invocation named and the run cannot use — a flag, a bound, a file, a
/// corpus cache that cannot be created, a corpus record or manifest the cache holds,
/// or a pinned recording that no longer matches its pin.
///
/// The type is the classification: a function whose failure means "this input is
/// unusable" returns `Result<_, Refused>`, so the marker rides on the error itself
/// and the entry point reads it once ([`super::exit_code`]) — no call site decides
/// it, and none can forget to.
#[derive(Debug, Clone)]
pub(crate) struct Refused(String);

impl Refused {
    /// The refusal, with the whole context chain as its message: that chain is what
    /// the console states.
    pub(crate) fn new(error: anyhow::Error) -> Self {
        Self(format!("{error:#}"))
    }
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// A function declared to return [`Refused`] states that everything it can fail with
/// is an unusable input, so its own `?` needs no decision.
impl From<anyhow::Error> for Refused {
    fn from(error: anyhow::Error) -> Self {
        Self::new(error)
    }
}

/// One input: the flag that names it and the environment variable that can
/// stand in for it. One table, so a flag's name and its variable cannot drift
/// apart and a new input is declared once.
#[derive(Clone, Copy)]
struct Spec {
    flag: &'static str,
    env: &'static str,
}

const MODELS: Spec = Spec {
    flag: "models",
    env: loader::MODELS_DIR_ENV,
};
const VOICE: Spec = Spec {
    flag: "voice",
    env: loader::VOICE_WAV_ENV,
};
const CORPUS: Spec = Spec {
    flag: "corpus",
    env: crate::corpus::ENV,
};
const REPORT: Spec = Spec {
    flag: "report",
    env: "MAHBOT_REPORT",
};
const LOCK: Spec = Spec {
    flag: "lock",
    env: "MAHBOT_LOCK",
};
const TIMEOUT: Spec = Spec {
    flag: "timeout",
    env: "MAHBOT_TIMEOUT_SECS",
};
const MAX_RSS: Spec = Spec {
    flag: "max-rss",
    env: "MAHBOT_MAX_RSS_MIB",
};
const ASR_BODY: Spec = Spec {
    flag: "asr-body",
    env: "MAHBOT_ASR_BODY",
};
const ASR_SOUND: Spec = Spec {
    flag: "asr-sound",
    env: "MAHBOT_ASR_SOUND",
};
const TTS_BODY: Spec = Spec {
    flag: "tts-body",
    env: "MAHBOT_TTS_BODY",
};
const TTS_SOUND: Spec = Spec {
    flag: "tts-sound",
    env: "MAHBOT_TTS_SOUND",
};
/// The phase selection. Unlike a path input, an empty value here — or one that is not
/// text — is an error, from the flag or from the variable; so is a bound's.
const PHASES: Spec = Spec {
    flag: "phases",
    env: "MAHBOT_BENCH_PHASES",
};
/// How many clone voices the synthetic side measures. A run that reaches the record
/// measures all [`crate::bench::sets::NUM_VERIFICATION_VOICES`]; fewer is a screening
/// run and writes no record. Like the other aids, an empty value is an error rather
/// than the default.
const VOICES: Spec = Spec {
    flag: "voices",
    env: "MAHBOT_BENCH_VOICES",
};

/// Every flag the bench accepts. Anything else is an error, so a typo cannot
/// silently fall back to an environment variable.
const SPECS: &[Spec] = &[
    MODELS, VOICE, CORPUS, REPORT, LOCK, TIMEOUT, MAX_RSS, ASR_BODY, ASR_SOUND, TTS_BODY,
    TTS_SOUND, PHASES, VOICES,
];

/// Which operation is resolving its inputs. The measurement names a report it
/// writes and is bounded by the defaults; the corpus build writes no report, takes
/// no measurement aid and is not a measurement at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Measure,
    Build,
}

impl Mode {
    /// The prefix the operation's own console lines carry, so a build's voice
    /// selection does not read as a measurement's.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Measure => "wake-word",
            Self::Build => "wake-word build",
        }
    }
}

/// The resolved inputs of one bench run.
#[derive(Debug, Clone)]
pub(crate) struct BenchConfig {
    /// The operation these inputs were resolved for: the measurement or the
    /// corpus build. The build writes no report and carries no wall-clock bound;
    /// its console lines say so.
    pub mode: Mode,
    /// The directory the role files are resolved against. Absent when all four
    /// role files were named explicitly.
    pub models: Option<PathBuf>,
    pub voice: PathBuf,
    pub corpus: PathBuf,
    /// The measurement's report file. Absent for the corpus build, which is not a
    /// measurement and writes no report.
    pub report: Option<PathBuf>,
    pub cache: PathBuf,
    pub lock: PathBuf,
    pub timeout_secs: u64,
    pub max_rss_mib: u64,
    pub asr_body: PathBuf,
    pub asr_sound: PathBuf,
    pub tts_body: PathBuf,
    pub tts_sound: PathBuf,
    /// Which of the three measurements this run takes. All three by default, and
    /// all three are what the benchmark itself takes: a run that takes a subset
    /// (to price a recall point without the real-audio sweep) reports `screened` and
    /// never writes the record.
    pub phases: Phases,
    /// How many clone voices the synthetic side measures, never more than the
    /// pinned pool fills ([`sets::NUM_VERIFICATION_VOICES`]). Fewer than all of
    /// them is a screening run and is labelled as such wherever the rows are
    /// printed.
    pub voices: usize,
    /// The swap-growth ceiling this run is under, resolved from the same
    /// environment the rest of the inputs came from — so a classification never
    /// reads a bound the run was not resolved under.
    pub swap_growth_mib: u64,
}

impl BenchConfig {
    /// Why this run is not the benchmark, in the order the reasons are checked. An
    /// empty list is the benchmark itself: all three phases, all eleven voices,
    /// inside the default wall-clock, resident and swap-growth bounds. The record
    /// only carries that run.
    pub(crate) fn screened_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if !self.phases.is_full() {
            reasons.push("a subset of the phases is measured");
        }
        if self.voices != sets::NUM_VERIFICATION_VOICES {
            reasons.push("a subset of the verification voices is measured");
        }
        reasons.extend(crate::run::bound_reasons(
            self.timeout_secs,
            self.max_rss_mib,
            self.swap_growth_mib,
        ));
        reasons
    }

    /// The process's own arguments and environment. A configuration that cannot be
    /// resolved is a refused input, so the whole resolution is classified once here.
    /// Arguments are read as the platform hands them over and lose only the bytes
    /// that are not UTF-8, so no argument can take the run down.
    pub(crate) fn from_env(mode: Mode) -> Result<Self, Refused> {
        Self::resolve(
            std::env::args_os()
                .skip(1)
                .map(|arg| arg.to_string_lossy().into_owned()),
            |name| std::env::var_os(name),
            mode,
        )
        .map_err(Refused::new)
    }

    /// The measurement's report. A resolved measurement always names one.
    pub(crate) fn report_path(&self) -> &Path {
        self.report
            .as_deref()
            .expect("a measurement names the report it writes")
    }

    /// The model directory for the run's opening line, or a note that the role
    /// files were named one by one.
    pub(crate) fn models_label(&self) -> String {
        self.models.as_ref().map_or_else(
            || "per-role files".to_string(),
            |path| path.display().to_string(),
        )
    }

    /// Resolve from an explicit argument list and environment lookup. Split out
    /// so the resolution rules are testable without touching either.
    pub(crate) fn resolve(
        argv: impl IntoIterator<Item = String>,
        env: impl Fn(&str) -> Option<OsString>,
        mode: Mode,
    ) -> Result<Self> {
        let flags = parse_flags(argv)?;
        // The build is not a measurement: it renders every clip of every voice,
        // writes no report and has no wall-clock bound, so the measurement aids
        // are refused rather than accepted and ignored.
        if mode == Mode::Build {
            let given_aids: Vec<String> = [REPORT, PHASES, VOICES, TIMEOUT]
                .into_iter()
                .filter(|spec| given(&flags, &env, *spec).is_some())
                .map(|spec| format!("--{} / {}", spec.flag, spec.env))
                .collect();
            if !given_aids.is_empty() {
                bail!(
                    "the corpus build is not a measurement — it renders every clip of every \
                     voice, writes no report and has no wall-clock bound — so it takes no {}",
                    given_aids.join(", ")
                );
            }
        }
        // Every input but the measurement aids and the three bounds reads an empty —
        // or non-text — value as absent: a required input then names itself, a
        // defaulted one takes its default. The aids and the bounds take the value as
        // given, so an empty or non-text flag *or* variable reaches its parser and is
        // refused there: `--phases ""`, a shell's `MAHBOT_BENCH_PHASES=`,
        // `--timeout ""` and a bound variable holding bytes must not resolve to the
        // defaults and rewrite the record.
        let get = |spec: Spec| {
            given(&flags, &env, spec)
                .and_then(|value| value.into_string().ok())
                .filter(|value| !value.is_empty())
        };

        let models = match get(MODELS) {
            Some(value) => {
                let path = absolute(&value)?;
                if !path.is_dir() {
                    bail!(
                        "the model directory {} is not a directory (--{} / {})",
                        path.display(),
                        MODELS.flag,
                        MODELS.env
                    );
                }
                Some(path)
            }
            None => None,
        };
        let role = |spec: Spec, role: Role, sound: bool| -> Result<PathBuf> {
            if let Some(value) = get(spec) {
                return absolute(&value);
            }
            let models = models.as_ref().ok_or_else(|| {
                anyhow!(
                    "--{} / {} is required unless --{} / {} names the directory holding both \
                     role files",
                    spec.flag,
                    spec.env,
                    MODELS.flag,
                    MODELS.env
                )
            })?;
            let (body, sound_name) = loader::pair_files(role)
                .ok_or_else(|| anyhow!("the loader has no `{}` pair", role.name()))?;
            Ok(models.join(if sound { sound_name } else { body }))
        };

        let voice = absolute(&required(
            get(VOICE),
            VOICE,
            "the recorded 16 kHz voice reference WAV",
        )?)?;
        if !voice.is_file() {
            bail!(
                "the voice WAV {} does not exist (--{} / {})",
                voice.display(),
                VOICE.flag,
                VOICE.env
            );
        }

        let corpus = absolute(&required(
            get(CORPUS),
            CORPUS,
            "the directory holding the pinned real-audio corpus",
        )?)?;
        if !corpus.is_dir() {
            bail!(
                "the corpus directory {} does not exist (--{} / {}); the real-audio phase cannot \
                 run without it",
                corpus.display(),
                CORPUS.flag,
                CORPUS.env
            );
        }

        let report = match mode {
            Mode::Measure => Some(absolute(&required(
                get(REPORT),
                REPORT,
                "the report file",
            )?)?),
            Mode::Build => None,
        };
        let cache = corpus_cache(&env)?;
        // One lock for every heavy run of this crate, from the same lookup as every
        // other input: an explicit path wins, otherwise the guard's shared default,
        // never "no lock".
        let lock = match get(LOCK) {
            Some(value) => absolute(&value)?,
            None => crate::guard::lock_path(&env)?,
        };

        let timeout_secs = bound(TIMEOUT, given(&flags, &env, TIMEOUT), DEFAULT_TIMEOUT_SECS)?;
        let max_rss_mib = bound(MAX_RSS, given(&flags, &env, MAX_RSS), DEFAULT_MAX_RSS_MIB)?;
        let phases = match given(&flags, &env, PHASES) {
            Some(value) => Phases::parse(&aid(value, PHASES)?)?,
            None => Phases::ALL,
        };
        let voices = match given(&flags, &env, VOICES) {
            Some(value) => voice_count(&aid(value, VOICES)?)?,
            None => sets::NUM_VERIFICATION_VOICES,
        };
        let swap_growth_mib = crate::guard::swap_growth_ceiling(&env).map_err(Refused::new)?;

        Ok(Self {
            mode,
            asr_body: role(ASR_BODY, Role::Recognition, false)?,
            asr_sound: role(ASR_SOUND, Role::Recognition, true)?,
            tts_body: role(TTS_BODY, Role::Synthesis, false)?,
            tts_sound: role(TTS_SOUND, Role::Synthesis, true)?,
            models,
            voice,
            corpus,
            report,
            cache,
            lock,
            timeout_secs,
            max_rss_mib,
            phases,
            voices,
            swap_growth_mib,
        })
    }
}

/// A voice count from a flag or its variable: at least one and at most the ten a
/// measurement is defined over. A typo or an out-of-range number is refused rather
/// than clamped — a clamped count would look like the eleven-voice measurement.
fn voice_count(text: &str) -> Result<usize> {
    let parsed: usize = text.trim().parse().with_context(|| {
        format!(
            "--{} / {} `{text}` is not a whole number",
            VOICES.flag, VOICES.env
        )
    })?;
    if parsed == 0 || parsed > sets::NUM_VERIFICATION_VOICES {
        bail!(
            "--{} / {} must be between 1 and {}",
            VOICES.flag,
            VOICES.env,
            sets::NUM_VERIFICATION_VOICES
        );
    }
    Ok(parsed)
}

/// Which of the three measurements a run takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Phases {
    pub recall: bool,
    pub non_phrase: bool,
    pub real_audio: bool,
}

impl Phases {
    /// All three measurements — the selection the benchmark itself takes.
    pub(crate) const ALL: Self = Self {
        recall: true,
        non_phrase: true,
        real_audio: true,
    };

    /// Whether the run takes all three measurements.
    pub(crate) fn is_full(&self) -> bool {
        *self == Self::ALL
    }

    /// `all`, or a comma list of `recall`, `non-phrase`, `real-audio`. Every name in
    /// the list is checked, `all` or not: a typo is an error wherever it sits.
    fn parse(text: &str) -> Result<Self> {
        let mut phases = Self {
            recall: false,
            non_phrase: false,
            real_audio: false,
        };
        let mut all = false;
        for name in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match name {
                "all" => all = true,
                "recall" => phases.recall = true,
                "non-phrase" => phases.non_phrase = true,
                "real-audio" => phases.real_audio = true,
                other => bail!(
                    "--{} / {} names no phase `{other}` (all, recall, non-phrase, real-audio)",
                    PHASES.flag,
                    PHASES.env
                ),
            }
        }
        if all {
            return Ok(Self::ALL);
        }
        if !phases.recall && !phases.non_phrase && !phases.real_audio {
            bail!("--{} / {} names no phase", PHASES.flag, PHASES.env);
        }
        Ok(phases)
    }
}

/// The synthesized corpus lives in one fixed, durable place — it is not an
/// input. A per-run path is how successive runs paid for the whole corpus again:
/// each was pointed at a freshly created directory, so the cache was always
/// empty.
///
/// `$XDG_CACHE_HOME/mahbot/wake-word-corpus`, else `~/.cache/mahbot/wake-word-corpus`.
/// Read through the same `env` lookup every other input uses, so a configuration
/// resolved in a test can never name the user's real cache.
fn corpus_cache(env: &impl Fn(&str) -> Option<OsString>) -> Result<PathBuf> {
    let root = match env("XDG_CACHE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match env("HOME") {
            Some(home) if !home.is_empty() => PathBuf::from(home).join(".cache"),
            _ => bail!(
                "neither XDG_CACHE_HOME nor HOME is set — there is no durable place for the \
                 synthesized corpus"
            ),
        },
    };
    Ok(root.join("mahbot").join("wake-word-corpus"))
}

/// A bound from its flag or its environment variable, through the guard's one
/// parser: the bench, the stand and the corpus run refuse a mistyped, empty or zero
/// bound with the same rule, and the figure the bench is steered by is the figure it
/// read. The value reaches the parser as given — a bound that is empty or not text is
/// refused rather than read as absent, which is what would let a run under a bound
/// nobody could read classify as complete and rewrite the record.
fn bound(spec: Spec, value: Option<OsString>, default: u64) -> Result<u64> {
    let source = format!("--{} / {}", spec.flag, spec.env);
    crate::guard::whole_number(value, default, &source)
}

/// One model file resolved to an absolute path, with its size and digest.
#[derive(Debug, Clone)]
pub(crate) struct Pin {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

impl Pin {
    /// The report cell for one file.
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "file": self
                .path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

/// The provenance of everything the measurement read.
#[derive(Debug, Clone)]
pub(crate) struct Inputs {
    pub asr_body: Pin,
    pub asr_sound: Pin,
    pub tts_body: Pin,
    pub tts_sound: Pin,
    pub voice: Pin,
}

impl Inputs {
    /// The `models` report cell: every model file by role, plus the voice WAV.
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "recognition": {
                "body": self.asr_body.json(),
                "sound": self.asr_sound.json(),
            },
            "synthesis": {
                "body": self.tts_body.json(),
                "sound": self.tts_sound.json(),
            },
            "voice_wav": self.voice.json(),
        })
    }

    /// The synthesis cache key's component that is not the request: both TTS
    /// files' digests, the voice reference's and the synthesis code itself, in a
    /// fixed order.
    ///
    /// The voice WAV is part of every prompt, so a key without it would serve
    /// clips in the old speaker under a report naming the new file. The code is
    /// in the key for the same reason: a prompt, a decode loop or a window change
    /// alters what the same text produces, and a stored clip must not outlive it.
    /// It is compiled in rather than read from the tree, so a binary run outside
    /// the checkout still knows the code it was built from.
    pub(crate) fn synthesis_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for digest in [
            &self.tts_body.sha256,
            &self.tts_sound.sha256,
            &self.voice.sha256,
        ] {
            hasher.update(digest.as_bytes());
            hasher.update([0u8]);
        }
        for source in SYNTHESIS_SOURCES {
            hash_code(&mut hasher, source);
            hasher.update([0u8]);
        }
        hex(&hasher.finalize())
    }
}

/// Hash the code lines of one source module, so a comment edit does not
/// invalidate the corpus.
///
/// Everything from the first column-0 `#[cfg(test)]` item on is skipped: a test
/// module never runs in a synthesising binary, and it is the last item in every
/// module in [`SYNTHESIS_SOURCES`]. A test-only edit therefore costs nothing,
/// while any change to code that can reach a sample does. The attribute has to
/// start in the first column to count, which is what tells a module-level item
/// apart from an attribute inside a function.
fn hash_code(hasher: &mut Sha256, source: &[u8]) {
    for line in source.split(|&byte| byte == b'\n') {
        let start = line
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())
            .unwrap_or(line.len());
        let code = &line[start..];
        if start == 0 && code.starts_with(b"#[cfg(test)]") {
            break;
        }
        if code.is_empty() || code.starts_with(b"//") {
            continue;
        }
        hasher.update(code);
        hasher.update(b"\n");
    }
}

/// Every source a synthesised clip's bytes depend on: the synthesis path, the
/// graph builders and the emitters under it, and the text and audio plumbing
/// that turns a phrase and a voice file into that path's inputs. A module added
/// to the path belongs here too.
const SYNTHESIS_SOURCES: &[&[u8]] = &[
    include_bytes!("../synth.rs"),
    include_bytes!("../graphs.rs"),
    include_bytes!("../emit.rs"),
    include_bytes!("../tokenizer.rs"),
    include_bytes!("../host.rs"),
    include_bytes!("../loader.rs"),
];

/// Stat and hash every input file. Errors name the file; a file the invocation named
/// and the run cannot use is a refused input, so the classification is made here once.
pub(crate) fn pin_inputs(cfg: &BenchConfig) -> Result<Inputs, Refused> {
    Ok(Inputs {
        asr_body: pin(&cfg.asr_body)?,
        asr_sound: pin(&cfg.asr_sound)?,
        tts_body: pin(&cfg.tts_body)?,
        tts_sound: pin(&cfg.tts_sound)?,
        voice: pin(&cfg.voice)?,
    })
}

/// A file's byte size and SHA-256, streamed so a multi-gigabyte model file is
/// not held in memory.
pub(crate) fn pin(path: &Path) -> Result<Pin> {
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("cannot stat {}", path.display()))?
        .len();
    Ok(Pin {
        path: path.to_path_buf(),
        bytes,
        sha256: sha256_file(path)?,
    })
}

fn parse_flags(argv: impl IntoIterator<Item = String>) -> Result<HashMap<String, String>> {
    let mut flags = HashMap::new();
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        // Cargo passes these to a `harness = false` target too; they are mode
        // markers, not inputs.
        if arg == "--bench" || arg == "--test" {
            continue;
        }
        let Some(body) = arg.strip_prefix("--") else {
            bail!(
                "unexpected argument `{arg}` — every argument is `--flag value` or `--flag=value`"
            );
        };
        let (name, value) = match body.split_once('=') {
            Some((name, value)) => (name.to_string(), value.to_string()),
            None => (
                body.to_string(),
                it.next()
                    .ok_or_else(|| anyhow!("`--{body}` needs a value"))?,
            ),
        };
        if !SPECS.iter().any(|spec| spec.flag == name) {
            bail!("unknown flag `--{name}`");
        }
        flags.insert(name, value);
    }
    Ok(flags)
}

/// The value a spec was given: the flag wins over its environment variable, and the
/// value is taken as written. What an empty or non-text value means is the caller's —
/// the path inputs read it as absent, the measurement aids and the bounds refuse it.
fn given(
    flags: &HashMap<String, String>,
    env: &impl Fn(&str) -> Option<OsString>,
    spec: Spec,
) -> Option<OsString> {
    flags
        .get(spec.flag)
        .map(OsString::from)
        .or_else(|| env(spec.env))
}

/// A measurement aid's value as text. A value that is not text is refused, not read
/// as absent: either would let a run that named nothing resolve to the shipped
/// defaults and rewrite the record.
fn aid(value: OsString, spec: Spec) -> Result<String> {
    value
        .into_string()
        .map_err(|_| anyhow!("--{} / {} is not UTF-8 text", spec.flag, spec.env))
}

fn required(value: Option<String>, spec: Spec, what: &str) -> Result<String> {
    value.ok_or_else(|| {
        anyhow!(
            "--{} / {} is required — it names {what}",
            spec.flag,
            spec.env
        )
    })
}

fn absolute(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    std::path::absolute(path).with_context(|| format!("cannot resolve the path {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "mahbot-wake-config-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    /// What resolution sees in a test: the two paths a configuration can default to —
    /// the corpus cache under `HOME`, the lock under `TMPDIR` — resolve inside the
    /// test's own tree, so a resolved test configuration can never name the user's real
    /// cache or lock.
    fn test_env(name: &str) -> Option<OsString> {
        match name {
            "HOME" => Some(OsString::from("/home/mahbot-test")),
            "TMPDIR" => Some(OsString::from("/home/mahbot-test/tmp")),
            _ => None,
        }
    }

    fn args(pairs: &[(&str, &str)]) -> Vec<String> {
        pairs
            .iter()
            .flat_map(|(flag, value)| [format!("--{flag}"), (*value).to_string()])
            .collect()
    }

    /// The synthesis key is code-only: comment edits and test modules stay out of
    /// it, while a change to code that can reach a sample moves it.
    #[test]
    fn the_synthesis_key_ignores_comments_and_test_modules() {
        let key = |source: &[u8]| {
            let mut hasher = Sha256::new();
            hash_code(&mut hasher, source);
            hex(&hasher.finalize())
        };
        let base = key(b"fn a() {}\n// note\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n");
        let commented =
            key(b"fn a() {}\n// other note\n#[cfg(test)]\nmod tests {\n    use std::x;\n}\n");
        assert_eq!(base, commented, "comments and test modules stay out");
        assert_ne!(
            base,
            key(b"fn a() { 1.0 }\n#[cfg(test)]\nmod tests {}\n"),
            "a code change moves the key"
        );
    }

    /// One temp directory holding a voice WAV and a corpus directory, and the
    /// argument list that names them plus the run's own outputs.
    struct Fixture {
        dir: PathBuf,
        args: Vec<String>,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let dir = temp_path(tag);
            std::fs::create_dir_all(dir.join("corpus")).expect("temp dirs");
            std::fs::write(dir.join("voice.wav"), b"RIFF").expect("voice");
            let path = |name: &str| dir.join(name).to_str().unwrap().to_string();
            let args = args(&[
                ("models", dir.to_str().unwrap()),
                ("voice", &path("voice.wav")),
                ("corpus", &path("corpus")),
                ("report", &path("report.json")),
                ("lock", &path("run.lock")),
            ]);
            Self { dir, args }
        }

        /// The same arguments with one flag removed.
        fn without(&self, flag: &str) -> Vec<String> {
            let mut args = self.args.clone();
            let at = args
                .iter()
                .position(|arg| arg == &format!("--{flag}"))
                .expect("the fixture sets every flag it removes");
            args.drain(at..at + 2);
            args
        }

        /// The same arguments with one flag replaced.
        fn with(&self, flag: &str, value: &str) -> Vec<String> {
            let mut args = self.args.clone();
            let at = args
                .iter()
                .position(|arg| arg == &format!("--{flag}"))
                .expect("the fixture sets every flag it replaces");
            args[at + 1] = value.to_string();
            args
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn a_missing_model_directory_names_the_flag_and_the_variable() {
        let err = BenchConfig::resolve(Vec::new(), test_env, Mode::Measure)
            .expect_err("no inputs at all");
        let text = format!("{err:#}");
        assert!(text.contains("--voice"), "{text}");
        assert!(text.contains(VOICE.env), "{text}");
    }

    #[test]
    fn a_missing_role_file_without_a_model_directory_names_both_flags() {
        let fixture = Fixture::new("role");
        let err = BenchConfig::resolve(fixture.without("models"), test_env, Mode::Measure)
            .expect_err("no directory, no role files");
        let text = format!("{err:#}");
        assert!(text.contains("--asr-body"), "{text}");
        assert!(text.contains(MODELS.env), "{text}");
    }

    #[test]
    fn a_missing_voice_names_the_flag_and_the_variable() {
        let fixture = Fixture::new("voice");
        let err = BenchConfig::resolve(
            fixture.with("voice", "/nope/voice.wav"),
            test_env,
            Mode::Measure,
        )
        .expect_err("the voice is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--voice"), "{text}");
        assert!(text.contains(VOICE.env), "{text}");
        assert!(text.contains("/nope/voice.wav"), "{text}");
    }

    #[test]
    fn a_missing_corpus_names_the_flag_and_the_variable() {
        let fixture = Fixture::new("corpus");
        let err = BenchConfig::resolve(
            fixture.with("corpus", "/nope/corpus"),
            test_env,
            Mode::Measure,
        )
        .expect_err("the corpus is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--corpus"), "{text}");
        assert!(text.contains(CORPUS.env), "{text}");
    }

    /// The report is a required input; the lock is not — it defaults to the one
    /// shared path every heavy run of this crate takes.
    #[test]
    fn the_report_is_a_required_input_and_the_lock_is_not() {
        let fixture = Fixture::new("report");
        let mut args = fixture.args.clone();
        let at = args.iter().position(|arg| arg == "--report").unwrap();
        args.drain(at..at + 2);
        let err =
            BenchConfig::resolve(args, test_env, Mode::Measure).expect_err("the report is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--report"), "{text}");
        assert!(text.contains(REPORT.env), "{text}");

        let cfg = BenchConfig::resolve(fixture.without("lock"), test_env, Mode::Measure)
            .expect("resolved");
        assert_eq!(
            cfg.lock,
            Path::new("/home/mahbot-test/tmp/mahbot-heavy-run.lock"),
            "the default lock is resolved from the same environment as every other input"
        );
    }

    #[test]
    fn a_path_that_does_not_exist_is_named() {
        let missing = temp_path("nope");
        let text = format!(
            "{:#}",
            BenchConfig::resolve(
                args(&[("models", missing.to_str().unwrap())]),
                test_env,
                Mode::Measure
            )
            .expect_err("the directory does not exist")
        );
        assert!(text.contains(missing.to_str().unwrap()), "{text}");
    }

    #[test]
    fn flags_win_over_the_environment_and_equals_form_is_accepted() {
        let fixture = Fixture::new("resolved");
        let mut argv = fixture.args.clone();
        argv.push("--timeout=1234".to_string());
        let cfg = BenchConfig::resolve(
            argv,
            |name| {
                if name == MODELS.env {
                    Some("/from/the/environment".into())
                } else {
                    test_env(name)
                }
            },
            Mode::Measure,
        )
        .expect("resolved");
        assert_eq!(cfg.models.as_deref(), Some(fixture.dir.as_path()));
        assert_eq!(cfg.timeout_secs, 1234);
        assert_eq!(
            cfg.report.as_deref(),
            Some(fixture.dir.join("report.json").as_path())
        );
        assert_eq!(
            cfg.cache,
            Path::new("/home/mahbot-test/.cache/mahbot/wake-word-corpus"),
            "the corpus cache is fixed and resolved from the same environment as every input"
        );
        assert_eq!(cfg.lock, fixture.dir.join("run.lock"));
    }

    #[test]
    fn all_four_role_files_remove_the_need_for_a_model_directory() {
        let fixture = Fixture::new("explicit");
        let mut argv = fixture.without("models");
        for flag in ["asr-body", "asr-sound", "tts-body", "tts-sound"] {
            argv.extend(args(&[(flag, "/models/file.gguf")]));
        }
        let cfg = BenchConfig::resolve(argv, test_env, Mode::Measure)
            .expect("the role files are all named");
        assert!(cfg.models.is_none());
        assert_eq!(cfg.asr_body, PathBuf::from("/models/file.gguf"));
        assert_eq!(cfg.models_label(), "per-role files");
    }

    /// The synthesis cache's key covers the voice reference, not only the pair:
    /// a changed `--voice` must not reuse clips in the old speaker while the
    /// report names the new file.
    #[test]
    fn the_synthesis_hash_covers_the_voice_reference() {
        let fixture = Fixture::new("voice-hash");
        let cfg = BenchConfig::resolve(fixture.args.clone(), test_env, Mode::Measure)
            .expect("the fixture");
        for path in [&cfg.asr_body, &cfg.asr_sound, &cfg.tts_body, &cfg.tts_sound] {
            std::fs::write(path, b"pair").expect("role file");
        }
        let before = pin_inputs(&cfg).expect("pins").synthesis_hash();
        std::fs::write(&cfg.voice, b"RIFF2").expect("voice");
        let after = pin_inputs(&cfg).expect("pins").synthesis_hash();
        assert_ne!(before, after, "the voice WAV is part of the key");
    }

    /// `cargo bench` appends its own mode marker to a `harness = false` target;
    /// it is not an input and must not be read as a flag missing its value.
    #[test]
    fn cargo_bench_mode_markers_are_ignored() {
        let fixture = Fixture::new("cargo-bench");
        let mut argv = fixture.args.clone();
        argv.push("--bench".to_string());
        let cfg = BenchConfig::resolve(argv, test_env, Mode::Measure).expect("--bench is a no-op");
        assert_eq!(cfg.timeout_secs, DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn an_unknown_flag_is_an_error() {
        let err = BenchConfig::resolve(
            vec!["--modles".to_string(), "x".to_string()],
            test_env,
            Mode::Measure,
        )
        .expect_err("unknown flag");
        assert!(format!("{err:#}").contains("--modles"));
    }

    #[test]
    fn a_screened_run_takes_only_the_phases_it_names() {
        let fixture = Fixture::new("phases");
        let full = BenchConfig::resolve(fixture.args.clone(), test_env, Mode::Measure)
            .expect("the fixture");
        assert!(full.phases.is_full());

        let mut argv = fixture.args.clone();
        argv.extend(args(&[("phases", "recall,non-phrase")]));
        let screened = BenchConfig::resolve(argv, test_env, Mode::Measure).expect("screened");
        assert!(screened.phases.recall && screened.phases.non_phrase);
        assert!(!screened.phases.real_audio);
        assert!(!screened.phases.is_full());

        let mut argv = fixture.args.clone();
        argv.extend(args(&[("phases", "recall,captions")]));
        let err = BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err("no such phase");
        assert!(format!("{err:#}").contains("captions"));

        // `all` does not end the check: a typo after it is still a typo.
        let mut argv = fixture.args.clone();
        argv.extend(args(&[("phases", "all,recalll")]));
        let err = BenchConfig::resolve(argv, test_env, Mode::Measure)
            .expect_err("no such phase after all");
        assert!(format!("{err:#}").contains("recalll"), "{err:#}");
    }

    /// An empty value is refused on every measurement aid, from the flag and from
    /// the environment: a run that overrode nothing must not resolve to the shipped
    /// defaults and rewrite the record. The refusal names both places, since the
    /// operator may have exported the variable and never passed the flag.
    #[test]
    fn an_empty_measurement_aid_is_an_error() {
        let fixture = Fixture::new("aids-empty");
        for (flag, env, wanted) in [
            (PHASES.flag, PHASES.env, "names no phase"),
            (VOICES.flag, VOICES.env, "is not a whole number"),
        ] {
            let mut argv = fixture.args.clone();
            argv.push(format!("--{flag}="));
            let err = BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err(flag);
            let message = format!("{err:#}");
            assert!(message.contains(wanted), "--{flag}: {message}");
            assert!(message.contains(&format!("--{flag} / {env}")), "{message}");

            // A shell that exports the name with no value is the same hazard.
            let err = BenchConfig::resolve(
                fixture.args.clone(),
                |name| {
                    if name == env {
                        Some(OsString::new())
                    } else {
                        test_env(name)
                    }
                },
                Mode::Measure,
            )
            .expect_err(env);
            let message = format!("{err:#}");
            assert!(message.contains(wanted), "{env}=: {message}");
            assert!(message.contains(&format!("--{flag} / {env}")), "{message}");
        }
    }

    /// A value that is not text is refused on the aids too: it would otherwise read
    /// as absent, which is the empty-value hazard by another spelling.
    #[cfg(unix)]
    #[test]
    fn a_non_text_measurement_aid_is_an_error() {
        use std::os::unix::ffi::OsStringExt;

        let fixture = Fixture::new("aids-bytes");
        for (flag, env) in [(PHASES.flag, PHASES.env), (VOICES.flag, VOICES.env)] {
            let bytes = OsString::from_vec(vec![0xff, b'=']);
            let err = BenchConfig::resolve(
                fixture.args.clone(),
                |name| {
                    if name == env {
                        Some(bytes.clone())
                    } else {
                        test_env(name)
                    }
                },
                Mode::Measure,
            )
            .expect_err(env);
            let message = format!("{err:#}");
            assert!(message.contains("is not UTF-8 text"), "{env}: {message}");
            assert!(message.contains(&format!("--{flag} / {env}")), "{message}");
        }
    }

    /// An empty value is refused by a bound, from the flag or from the variable: a
    /// run under a bound nobody could read must not take the default and then
    /// classify as complete and rewrite the record. The stand and the corpus run read
    /// the same variables through the same parser, so the answer is the same there.
    #[test]
    fn an_empty_bound_is_refused_rather_than_defaulted() {
        let fixture = Fixture::new("bounds-empty");
        for (flag, env) in [(TIMEOUT.flag, TIMEOUT.env), (MAX_RSS.flag, MAX_RSS.env)] {
            let mut argv = fixture.args.clone();
            argv.push(format!("--{flag}="));
            let err =
                BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err("an empty flag");
            let message = format!("{err:#}");
            assert!(
                message.contains(&format!("--{flag} / {env}")),
                "{env}: {message}"
            );
            assert!(
                message.contains("is not a whole number"),
                "{env}: {message}"
            );

            let empty = |name: &str| {
                if name == env {
                    Some(OsString::new())
                } else {
                    test_env(name)
                }
            };
            let err = BenchConfig::resolve(fixture.args.clone(), empty, Mode::Measure)
                .expect_err("an empty variable");
            let message = format!("{err:#}");
            assert!(
                message.contains(&format!("--{flag} / {env}")),
                "{env}: {message}"
            );
            assert!(
                message.contains("is not a whole number"),
                "{env}: {message}"
            );
        }
        // The swap-growth ceiling is the third bound, and reads the same way.
        let err = BenchConfig::resolve(
            fixture.args.clone(),
            |name| {
                if name == "MAHBOT_STAND_MAX_SWAP_MIB" {
                    Some(OsString::new())
                } else {
                    test_env(name)
                }
            },
            Mode::Measure,
        )
        .expect_err("an empty swap bound");
        assert!(
            format!("{err:#}").contains("MAHBOT_STAND_MAX_SWAP_MIB"),
            "{err:#}"
        );
    }

    /// An empty value on every other input is absent: a required one names itself.
    #[test]
    fn an_empty_input_is_absent_outside_the_bounds_and_aids() {
        let fixture = Fixture::new("inputs-empty");
        let mut argv = fixture.args.clone();
        argv.push("--voice=".to_string());
        let err = BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err("an empty voice");
        assert!(
            format!("{err:#}").contains("is required"),
            "an empty path reads as not given: {err:#}"
        );
    }

    /// The corpus build is not a measurement: it names no report, takes no
    /// measurement aid (all of which would be accepted and ignored), and a
    /// measurement still requires its report.
    #[test]
    fn the_build_takes_no_measurement_aid() {
        let fixture = Fixture::new("mode");
        let err = BenchConfig::resolve(fixture.without("report"), test_env, Mode::Measure)
            .expect_err("a measurement names its report");
        assert!(format!("{err:#}").contains("--report"), "{err:#}");
        let cfg = BenchConfig::resolve(fixture.without("report"), test_env, Mode::Build)
            .expect("the build");
        assert!(cfg.report.is_none());

        for argv in [
            args(&[("report", "/tmp/report.json")]),
            args(&[("voices", "3")]),
            args(&[("phases", "recall")]),
            args(&[("timeout", "600")]),
        ] {
            let mut full = fixture.without("report");
            full.extend(argv.clone());
            let err = BenchConfig::resolve(full, test_env, Mode::Build).expect_err("an aid");
            assert!(format!("{err:#}").contains("takes no"), "{argv:?}: {err:#}");
        }
        // The memory ceiling is a guard the build really carries, so it is a bound
        // the build may be given — unlike the measurement's wall-clock stop.
        let mut full = fixture.without("report");
        full.extend(args(&[("max-rss", "8192")]));
        assert_eq!(
            BenchConfig::resolve(full, test_env, Mode::Build)
                .expect("a build may raise its memory ceiling")
                .max_rss_mib,
            8_192
        );
    }

    /// A raised swap-growth ceiling is a screening reason like the other bounds, so
    /// no run can be recorded under one — and the ceiling is read from the same
    /// environment the run was resolved from, not from the live process.
    #[test]
    fn a_raised_swap_bound_is_screened() {
        assert!(crate::run::bound_reasons(0, 0, crate::guard::MAX_SWAP_GROWTH_MIB).is_empty());
        assert!(!crate::run::bound_reasons(0, 0, crate::guard::MAX_SWAP_GROWTH_MIB + 1).is_empty());

        let fixture = Fixture::new("swap");
        let raised = |name: &str| {
            if name == "MAHBOT_STAND_MAX_SWAP_MIB" {
                Some(OsString::from("8192"))
            } else {
                test_env(name)
            }
        };
        let cfg = BenchConfig::resolve(fixture.args.clone(), raised, Mode::Measure)
            .expect("a run under a raised swap bound");
        assert_eq!(cfg.swap_growth_mib, 8_192);
        assert_eq!(
            cfg.screened_reasons(),
            vec!["the swap-growth bound is larger than the default"]
        );
    }

    #[test]
    fn zero_timeout_is_an_error() {
        let fixture = Fixture::new("timeout");
        let mut argv = fixture.args.clone();
        argv.push("--timeout=0".to_string());
        let err = BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err("zero timeout");
        assert!(format!("{err:#}").contains("--timeout"));
    }

    /// A voice count outside the pool's own range is refused, never clamped: a
    /// clamped count would look like the eleven-voice measurement.
    #[test]
    fn a_voice_count_outside_the_pool_is_an_error() {
        let fixture = Fixture::new("voices");
        for value in ["0", "11", "ten"] {
            let mut argv = fixture.args.clone();
            argv.push(format!("--voices={value}"));
            let err = BenchConfig::resolve(argv, test_env, Mode::Measure).expect_err(value);
            assert!(format!("{err:#}").contains("--voices"), "{value}: {err:#}");
        }
    }

    /// The benchmark is the shipped point over all three phases, all eleven
    /// voices, inside the defaults — and nothing else is.
    #[test]
    fn only_the_full_run_inside_the_defaults_is_the_benchmark() {
        let fixture = Fixture::new("screened");
        let cfg = BenchConfig::resolve(fixture.args.clone(), test_env, Mode::Measure)
            .expect("the defaults");
        assert!(
            cfg.screened_reasons().is_empty(),
            "{:?}",
            cfg.screened_reasons()
        );

        for argv in [
            args(&[("voices", "3")]),
            args(&[("phases", "recall")]),
            args(&[("timeout", "7200")]),
            args(&[("max-rss", "8192")]),
        ] {
            let mut full = fixture.args.clone();
            full.extend(argv.clone());
            let cfg = BenchConfig::resolve(full, test_env, Mode::Measure).expect("a screening run");
            assert_eq!(cfg.screened_reasons().len(), 1, "{argv:?}");
        }

        // A tighter bound is still the benchmark: it can only stop the run sooner.
        let mut full = fixture.args.clone();
        full.extend(args(&[("timeout", "600")]));
        let cfg = BenchConfig::resolve(full, test_env, Mode::Measure).expect("a tighter bound");
        assert!(
            cfg.screened_reasons().is_empty(),
            "{:?}",
            cfg.screened_reasons()
        );
    }
}
