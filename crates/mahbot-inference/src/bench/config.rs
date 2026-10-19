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
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

use crate::loader::{self, Role};
use crate::measurements::{hex, sha256_file};

use crate::guard::{DEFAULT_MAX_RSS_MIB, DEFAULT_TIMEOUT_SECS};

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
    env: "MAHBOT_FAPH_CORPUS",
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

/// Every flag the bench accepts. Anything else is an error, so a typo cannot
/// silently fall back to an environment variable.
const SPECS: &[Spec] = &[
    MODELS, VOICE, CORPUS, REPORT, LOCK, TIMEOUT, MAX_RSS, ASR_BODY, ASR_SOUND, TTS_BODY, TTS_SOUND,
];

/// The resolved inputs of one bench run.
#[derive(Debug, Clone)]
pub(crate) struct BenchConfig {
    /// The directory the role files are resolved against. Absent when all four
    /// role files were named explicitly.
    pub models: Option<PathBuf>,
    pub voice: PathBuf,
    pub corpus: PathBuf,
    pub report: PathBuf,
    pub cache: PathBuf,
    pub lock: PathBuf,
    pub timeout_secs: u64,
    pub max_rss_mib: u64,
    pub asr_body: PathBuf,
    pub asr_sound: PathBuf,
    pub tts_body: PathBuf,
    pub tts_sound: PathBuf,
}

impl BenchConfig {
    /// The process's own arguments and environment.
    pub(crate) fn from_env() -> Result<Self> {
        Self::resolve(std::env::args().skip(1), |name| std::env::var(name).ok())
    }

    /// The invocation, with every input it took, for the provenance block: the
    /// report this run wrote is the thing a later reader has to be able to
    /// repeat. The four role files are recorded by name rather than the directory
    /// they came from, so the line replays whether the run was given `--models`
    /// or the files one by one; paths under `$HOME` are shortened to `~`.
    pub(crate) fn command_line(&self) -> String {
        let short = crate::measurements::short_path;
        format!(
            "cargo bench -p mahbot-inference --features bench --bench wake_word -- \
             --asr-body {} --asr-sound {} --tts-body {} --tts-sound {} --voice {} \
             --corpus {} --report {} --lock {} --timeout {} --max-rss {}",
            short(&self.asr_body),
            short(&self.asr_sound),
            short(&self.tts_body),
            short(&self.tts_sound),
            short(&self.voice),
            short(&self.corpus),
            short(&self.report),
            short(&self.lock),
            self.timeout_secs,
            self.max_rss_mib,
        )
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
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let flags = parse_flags(argv)?;
        let get = |spec: Spec| pick(&flags, &env, spec);

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

        let report = absolute(&required(get(REPORT), REPORT, "the report file")?)?;
        let cache = corpus_cache()?;
        // One lock for every heavy run of this crate: an explicit path wins,
        // otherwise the shared default, never "no lock".
        let lock = match get(LOCK) {
            Some(value) => absolute(&value)?,
            None => crate::guard::lock_path(),
        };

        let timeout_secs = whole_number(get(TIMEOUT), DEFAULT_TIMEOUT_SECS, TIMEOUT)?;
        let max_rss_mib = whole_number(get(MAX_RSS), DEFAULT_MAX_RSS_MIB, MAX_RSS)?;

        Ok(Self {
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
        })
    }
}

/// The synthesized corpus lives in one fixed, durable place — it is not an
/// input. A per-run path is how successive runs paid for the whole corpus again:
/// each was pointed at a freshly created directory, so the cache was always
/// empty.
///
/// `$XDG_CACHE_HOME/mahbot/wake-word-corpus`, else `~/.cache/mahbot/wake-word-corpus`.
fn corpus_cache() -> Result<PathBuf> {
    let root = match std::env::var_os("XDG_CACHE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match std::env::var_os("HOME") {
            Some(home) if !home.is_empty() => PathBuf::from(home).join(".cache"),
            _ => bail!(
                "neither XDG_CACHE_HOME nor HOME is set — there is no durable place for the \
                 synthesized corpus"
            ),
        },
    };
    Ok(root.join("mahbot").join("wake-word-corpus"))
}

/// A positive whole number from a flag or its environment variable, or the
/// documented default.
fn whole_number(value: Option<String>, default: u64, spec: Spec) -> Result<u64> {
    let parsed = match value {
        Some(text) => text.parse::<u64>().with_context(|| {
            format!(
                "--{} / {} `{text}` is not a whole number",
                spec.flag, spec.env
            )
        })?,
        None => default,
    };
    if parsed == 0 {
        bail!("--{} / {} must be at least 1", spec.flag, spec.env);
    }
    Ok(parsed)
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
            "path": self.path.to_string_lossy(),
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

/// Stat and hash every input file. Errors name the file.
pub(crate) fn pin_inputs(cfg: &BenchConfig) -> Result<Inputs> {
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

/// A flag wins over its environment variable.
fn pick(
    flags: &HashMap<String, String>,
    env: &impl Fn(&str) -> Option<String>,
    spec: Spec,
) -> Option<String> {
    flags
        .get(spec.flag)
        .cloned()
        .or_else(|| env(spec.env))
        .filter(|value| !value.is_empty())
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
        let err = BenchConfig::resolve(Vec::new(), |_| None).expect_err("no inputs at all");
        let text = format!("{err:#}");
        assert!(text.contains("--voice"), "{text}");
        assert!(text.contains(VOICE.env), "{text}");
    }

    #[test]
    fn a_missing_role_file_without_a_model_directory_names_both_flags() {
        let fixture = Fixture::new("role");
        let err = BenchConfig::resolve(fixture.without("models"), |_| None)
            .expect_err("no directory, no role files");
        let text = format!("{err:#}");
        assert!(text.contains("--asr-body"), "{text}");
        assert!(text.contains(MODELS.env), "{text}");
    }

    #[test]
    fn a_missing_voice_names_the_flag_and_the_variable() {
        let fixture = Fixture::new("voice");
        let err = BenchConfig::resolve(fixture.with("voice", "/nope/voice.wav"), |_| None)
            .expect_err("the voice is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--voice"), "{text}");
        assert!(text.contains(VOICE.env), "{text}");
        assert!(text.contains("/nope/voice.wav"), "{text}");
    }

    #[test]
    fn a_missing_corpus_names_the_flag_and_the_variable() {
        let fixture = Fixture::new("corpus");
        let err = BenchConfig::resolve(fixture.with("corpus", "/nope/corpus"), |_| None)
            .expect_err("the corpus is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--corpus"), "{text}");
        assert!(text.contains(CORPUS.env), "{text}");
    }

    /// The recorded invocation must replay as written: the four role files are
    /// what the run actually resolved, so they are what the line names — a
    /// directory-based line would fail with "`--models` is required" once the
    /// reader tried it from a checkout that has no such directory.
    #[test]
    fn the_recorded_command_names_the_files_the_run_used() {
        let fixture = Fixture::new("command");
        let cfg = BenchConfig::resolve(fixture.args.clone(), |_| None).expect("resolved");
        let line = cfg.command_line();
        for flag in ["--asr-body", "--asr-sound", "--tts-body", "--tts-sound"] {
            assert!(line.contains(flag), "{flag} is missing from: {line}");
        }
        assert!(
            !line.contains("--models"),
            "the line must not depend on a directory: {line}"
        );
        for path in [&cfg.asr_body, &cfg.tts_sound, &cfg.voice, &cfg.corpus] {
            let shown = crate::measurements::short_path(path);
            assert!(line.contains(&shown), "{shown} is missing from: {line}");
        }
    }

    /// The report is a required input; the lock is not — it defaults to the one
    /// shared path every heavy run of this crate takes.
    #[test]
    fn the_report_is_a_required_input_and_the_lock_is_not() {
        let fixture = Fixture::new("report");
        let mut args = fixture.args.clone();
        let at = args.iter().position(|arg| arg == "--report").unwrap();
        args.drain(at..at + 2);
        let err = BenchConfig::resolve(args, |_| None).expect_err("the report is missing");
        let text = format!("{err:#}");
        assert!(text.contains("--report"), "{text}");
        assert!(text.contains(REPORT.env), "{text}");

        let cfg = BenchConfig::resolve(fixture.without("lock"), |_| None).expect("resolved");
        assert_eq!(cfg.lock, crate::guard::lock_path());
    }

    #[test]
    fn a_path_that_does_not_exist_is_named() {
        let missing = temp_path("nope");
        let text = format!(
            "{:#}",
            BenchConfig::resolve(args(&[("models", missing.to_str().unwrap())]), |_| None)
                .expect_err("the directory does not exist")
        );
        assert!(text.contains(missing.to_str().unwrap()), "{text}");
    }

    #[test]
    fn flags_win_over_the_environment_and_equals_form_is_accepted() {
        let fixture = Fixture::new("resolved");
        let mut argv = fixture.args.clone();
        argv.push("--timeout=1234".to_string());
        let cfg = BenchConfig::resolve(argv, |name| {
            (name == MODELS.env).then(|| "/from/the/environment".to_string())
        })
        .expect("resolved");
        assert_eq!(cfg.models.as_deref(), Some(fixture.dir.as_path()));
        assert_eq!(cfg.timeout_secs, 1234);
        assert_eq!(cfg.report, fixture.dir.join("report.json"));
        assert!(
            cfg.cache.ends_with("mahbot/wake-word-corpus"),
            "the corpus cache is fixed: {}",
            cfg.cache.display()
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
        let cfg = BenchConfig::resolve(argv, |_| None).expect("the role files are all named");
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
        let cfg = BenchConfig::resolve(fixture.args.clone(), |_| None).expect("the fixture");
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
        let cfg = BenchConfig::resolve(argv, |_| None).expect("--bench is a no-op");
        assert_eq!(cfg.timeout_secs, DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn an_unknown_flag_is_an_error() {
        let err = BenchConfig::resolve(vec!["--modles".to_string(), "x".to_string()], |_| None)
            .expect_err("unknown flag");
        assert!(format!("{err:#}").contains("--modles"));
    }

    #[test]
    fn zero_timeout_is_an_error() {
        let fixture = Fixture::new("timeout");
        let mut argv = fixture.args.clone();
        argv.push("--timeout=0".to_string());
        let err = BenchConfig::resolve(argv, |_| None).expect_err("zero timeout");
        assert!(format!("{err:#}").contains("--timeout"));
    }
}
