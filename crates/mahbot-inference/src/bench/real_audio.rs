// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The real-audio phase: the pinned 33-file corpus, fed in order through one
//! continuous listening stream, and the false accepts per hour it produces.
//!
//! One stream, and nothing carries between two runs of it: the detector, the
//! adaptive state and the 3 s audio-clock cooldown carry across every file, with
//! two seconds of digital silence between files (which closes the segment without
//! touching the detector). A fire is one event: the stream's own audio-clock
//! cooldown already suppresses a second one inside the same 3 s of audio.
//!
//! The stream embeds its windows in groups of [`listen::GROUP`], so a file costs
//! one encoder pass per group of windows rather than one per window. The group
//! size is a fact of the run and is reported with it.
//!
//! The wall-clock stop is checked on every detection chunk and at every file; the
//! resident set and the two system figures are checked at every file, and the
//! guard thread covers a blocked device call. A stop inside the phase still leaves
//! what the files fed so far measured — files, audio hours, VAD-active hours and
//! events are taken from the same stream, so the figures in a cut report describe
//! one population.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::listen::Embedder;
use crate::loader::Pair;
use crate::wake::Enrollment;
use crate::{host, listen};

use super::config::BenchConfig;
use super::detect::{self, DetectionStream};
use super::sets;
use crate::guard::{Deadline, MemoryCeiling};

/// The phase's totals: what the stream fed and what it fired on.
#[derive(Default)]
struct Totals {
    files_fed: usize,
    audio_secs: f64,
    speech_secs: f64,
    noise_secs: f64,
    vad_secs: f64,
    events: usize,
}

impl Totals {
    /// Credit one file with the seconds of its audio that were actually fed: the
    /// whole file on a complete feed, its prefix on a cut. A stop that arrives
    /// before the file's first chunk credits nothing, so the file is not counted.
    fn credit(&mut self, path: &str, secs: f64) {
        if secs <= 0.0 {
            return;
        }
        self.files_fed += 1;
        self.audio_secs += secs;
        if is_speech(path) {
            self.speech_secs += secs;
        } else {
            self.noise_secs += secs;
        }
    }
}

/// The pool a pinned file belongs to: the set partitions into `speech/` and
/// `noise/`, so a path that is not speech is noise, here and in the report's
/// per-pool file counts.
fn is_speech(path: &str) -> bool {
    path.starts_with("speech/")
}

/// Everything the report states about the real-audio phase.
pub(crate) struct RealAudio {
    pub files_total: usize,
    pub speech_files: usize,
    pub noise_files: usize,
    pub corpus: PathBuf,
    pub files_fed: usize,
    pub audio_hours: f64,
    pub speech_hours: f64,
    pub noise_hours: f64,
    pub vad_active_hours: f64,
    pub wall_secs: f64,
    /// Windows in the largest group the stream resolved, how many encoder passes
    /// its windows took, and how many of those carried fewer windows than the
    /// capacity — the pipeline is built for `listen::GROUP`; a phase cut short
    /// may never fill one.
    pub max_group: usize,
    pub encoder_passes: usize,
    pub partial_passes: usize,
    /// One listener's footprint, for the memory the phase actually held.
    pub listener_bytes: u64,
    pub events: usize,
    /// Why the phase stopped early, when it did; the figures above are then what
    /// the files fed so far measured.
    pub cut: Option<String>,
    pub failures: Vec<String>,
}

impl RealAudio {
    /// The report section. A per-hour rate is produced only from a positive
    /// denominator; where one is zero the key is absent and `omitted` names it,
    /// so a degenerate rate is never serialised as a null.
    pub(crate) fn json(&self) -> Value {
        let status = if self.cut.is_some() || !self.failures.is_empty() {
            "partial"
        } else {
            "ran"
        };
        let mut fa = serde_json::Map::new();
        fa.insert("events".into(), json!(self.events));
        let mut omitted = Vec::new();
        for (key, hours) in [
            ("fa_per_hour_audio", self.audio_hours),
            ("fa_per_hour_vad_active", self.vad_active_hours),
        ] {
            if hours > 0.0 {
                fa.insert(key.into(), json!(self.events as f64 / hours));
            } else {
                omitted.push(key);
            }
        }
        if !omitted.is_empty() {
            fa.insert(
                "omitted".into(),
                json!(format!(
                    "{} — the denominator measured no time, so no rate is reported",
                    omitted.join(", ")
                )),
            );
        }
        fa.insert(
            "basis_note".into(),
            json!(
                "One continuous stream over the pinned files in their pinned order: the \
                 detector, the adaptive state and the 3 s audio-clock cooldown carry across \
                 files, and two seconds of digital silence separate them. A fire is one event; \
                 the cooldown already suppresses a second one inside the same 3 s of audio. \
                 The denominators are raw audio hours and VAD-active hours over that one stream. \
                 Windows are encoded in groups of sixteen, so after a fire the detector has \
                 already run up to one group (2.56 s) of the hops that follow it, and those \
                 hops count in the VAD-active denominator: the bound is stated, not corrected."
            ),
        );
        json!({
            "status": status,
            "subset": {
                "files_total": self.files_total,
                "speech_files": self.speech_files,
                "noise_files": self.noise_files,
                "corpus": self.corpus.to_string_lossy(),
                "manifest": {"repo": sets::REAL_AUDIO_REPO, "revision": sets::REAL_AUDIO_REVISION},
                "selection_heuristic": "longest librivox speech files up to a target + all \
                                        us-gov speech files + a fixed noise selection (longest \
                                        free-sound + sound-bible clips) — pinned as a bench \
                                        constant with each file's SHA-256 and size",
            },
            "feed": {
                "streams": 1,
                "windows_per_pass": self.max_group,
                "group_capacity": listen::GROUP,
                "encoder_passes": self.encoder_passes,
                "partial_passes": self.partial_passes,
                "listener_bytes": self.listener_bytes,
                "files_fed": self.files_fed,
                "audio_hours_fed": self.audio_hours,
                "speech_hours_fed": self.speech_hours,
                "noise_hours_fed": self.noise_hours,
                "vad_active_hours_fed": self.vad_active_hours,
                "wall_secs": self.wall_secs,
                "cut": self.cut,
                "failures": self.failures,
            },
            "fa": Value::Object(fa),
        })
    }
}

/// The phase's figures, and why it was cut short when it was: a stop inside the
/// phase still leaves what the files fed so far measured.
pub(crate) struct Cut {
    pub audio: RealAudio,
    pub error: Option<anyhow::Error>,
}

/// Feed the pinned subset and return the phase's totals. The corpus is verified
/// before the run's expensive phases (see [`super::measure`]), not here.
pub(crate) fn run(
    cfg: &BenchConfig,
    pair: &Arc<Pair>,
    enrollment: &Enrollment,
    deadline: Deadline,
    ceiling: MemoryCeiling,
) -> Cut {
    let started = std::time::Instant::now();
    let subset = sets::REAL_AUDIO_FILES;
    let mut totals = Totals::default();
    let mut failures = Vec::new();
    let mut cut: Option<anyhow::Error> = None;
    let mut listener_bytes = 0u64;

    // The stream outlives the listener: it is built whether or not the front end
    // compiles, so its group facts stay readable even when the phase is cut.
    let mut stream = DetectionStream::grouped(enrollment.clone());
    let listener = match listen::GroupFront::build(pair, listen::MEL_FRAMES, listen::GROUP) {
        Ok(listener) => {
            listener_bytes = listener.footprint_bytes();
            Some(listener)
        }
        Err(error) => {
            cut = Some(error.context("the real-audio listener did not compile"));
            None
        }
    };

    if let Some(mut listener) = listener {
        let gap = vec![0f32; sets::FILE_GAP_SAMPLES];
        // A panic in the stream is one named failure with what was fed so far,
        // not a process that dies holding the lock.
        let fed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            feed_files(
                subset,
                &cfg.corpus,
                &mut listener,
                &mut stream,
                &gap,
                deadline,
                ceiling,
                &mut totals,
            )
        }));
        match fed {
            Ok(Ok(())) => {}
            Ok(Err(error)) => cut = Some(error),
            Err(_) => {
                failures.push("the listening stream panicked".to_string());
                cut = Some(anyhow!("the real-audio stream panicked"));
            }
        }
    }

    if cut.is_none() && totals.files_fed == 0 {
        let detail = if failures.is_empty() {
            String::new()
        } else {
            format!(": {}", failures.join("; "))
        };
        cut = Some(anyhow!(
            "the real-audio phase fed no files — nothing was measured{detail}"
        ));
    }

    let audio = RealAudio {
        files_total: subset.len(),
        speech_files: subset.iter().filter(|(path, _, _)| is_speech(path)).count(),
        noise_files: subset
            .iter()
            .filter(|(path, _, _)| !is_speech(path))
            .count(),
        corpus: cfg.corpus.clone(),
        files_fed: totals.files_fed,
        audio_hours: totals.audio_secs / 3600.0,
        speech_hours: totals.speech_secs / 3600.0,
        noise_hours: totals.noise_secs / 3600.0,
        vad_active_hours: totals.vad_secs / 3600.0,
        wall_secs: started.elapsed().as_secs_f64(),
        max_group: stream.max_group(),
        encoder_passes: stream.groups(),
        partial_passes: stream.partial_groups(),
        listener_bytes,
        events: totals.events,
        cut: cut.as_ref().map(|error| format!("{error:#}")),
        failures,
    };
    Cut { audio, error: cut }
}

/// Size and SHA-256 of every pinned file. Run in the input phase, before the
/// expensive part: a bad pin stops the run in its first seconds, not after the
/// corpus synthesis. A mismatch is an error naming the file.
pub(crate) fn verify(corpus: &Path, subset: &[(&str, &str, u64)]) -> Result<()> {
    for (path, sha256, pinned) in subset {
        let file = corpus.join(path);
        let bytes = std::fs::metadata(&file)
            .with_context(|| format!("cannot stat the pinned corpus file {}", file.display()))?
            .len();
        if bytes != *pinned {
            bail!(
                "the pinned corpus file {} is {bytes} bytes, not the pinned {pinned} bytes",
                file.display()
            );
        }
        let actual = crate::measurements::sha256_file(&file)?;
        if actual != *sha256 {
            bail!(
                "the pinned corpus file {} does not match its SHA-256 pin ({actual}, expected \
                 {sha256})",
                file.display()
            );
        }
    }
    Ok(())
}

/// The one stream: the pinned files in order, each followed by the silence gap,
/// under the run's stop and memory ceiling at every file.
fn feed_files(
    files: &[(&str, &str, u64)],
    corpus: &Path,
    listener: &mut impl Embedder,
    stream: &mut DetectionStream,
    gap: &[f32],
    deadline: Deadline,
    ceiling: MemoryCeiling,
    totals: &mut Totals,
) -> Result<()> {
    for (path, _, _) in files {
        deadline.check("real audio")?;
        ceiling.check("real audio")?;
        let file = corpus.join(path);
        let name = file
            .to_str()
            .ok_or_else(|| anyhow!("the corpus path {} is not UTF-8", file.display()))?;
        let samples = host::wav_mono(name, listen::SAMPLE_RATE, 0, usize::MAX)
            .with_context(|| format!("cannot decode {}", file.display()))?;
        let before = stream.samples_fed();
        let fed = stream.feed(&samples, listener, deadline, "real audio");
        // Credited after the feed, from the audio it took, before the error leaves:
        // what a cut report's rate divides by.
        let secs = (stream.samples_fed() - before) as f64 / f64::from(listen::SAMPLE_RATE);
        take_measured(stream, totals);
        totals.credit(path, secs);
        fed?;
        let gap_fed = stream.feed(gap, listener, deadline, "real audio");
        take_measured(stream, totals);
        gap_fed?;
    }

    Ok(())
}

/// What the stream measured since the last read — fires and VAD-active seconds —
/// taken together, so a cut report's rate divides one population.
fn take_measured(stream: &mut DetectionStream, totals: &mut Totals) {
    totals.events += stream.take_fires();
    totals.vad_secs +=
        stream.take_vad_hops() as f64 * detect::HOP_LENGTH as f64 / f64::from(listen::SAMPLE_RATE);
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;

    use super::{Totals, feed_files};
    use crate::bench::detect::DetectionStream;
    use crate::bench::sets;
    use crate::guard::{Deadline, MemoryCeiling};
    use crate::listen::Embedder;
    use crate::{listen, wake};

    /// A synthesized corpus clip: 16 kHz mono PCM16.
    fn wav(path: &Path, secs: f64) {
        let frames = (secs * f64::from(listen::SAMPLE_RATE)) as usize;
        let data = frames as u32 * 2;
        let mut bytes = Vec::with_capacity(44 + data as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&listen::SAMPLE_RATE.to_le_bytes());
        bytes.extend_from_slice(&(listen::SAMPLE_RATE * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data.to_le_bytes());
        for i in 0..frames {
            let tone = (i as f32 * 440.0 / listen::SAMPLE_RATE as f32 % 1.0) - 0.5;
            bytes.extend_from_slice(&((tone * 16_000.0) as i16).to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write the synthetic clip");
    }

    /// Answers every window with one fixed embedding: this test measures the
    /// feed's accounting, not what the windows score.
    struct Fixed;

    impl Embedder for Fixed {
        fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0, 0.0]; windows.len()])
        }
    }

    fn stream() -> DetectionStream {
        let utterances = vec![vec![1.0f32, 0.0]; wake::MIN_ENROLLMENT_UTTERANCES];
        let enrollment = wake::Enrollment::build(
            "wake".into(),
            &utterances,
            wake::Calibration::default(),
            &[],
        )
        .expect("an enrolment of identical utterances");
        DetectionStream::warm(enrollment)
    }

    /// The phase credits each file with the audio its feed took, into the pool its
    /// path names. The defect this pins was crediting the file before feeding it,
    /// so the arithmetic here is the call site's — the stream's cursor around the
    /// feed — not [`Totals::credit`]'s on hand-written seconds.
    #[test]
    fn each_file_is_credited_with_the_audio_its_feed_took() {
        let dir = std::env::temp_dir().join(format!("mahbot-real-audio-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("speech")).expect("temp corpus");
        std::fs::create_dir_all(dir.join("noise")).expect("temp corpus");
        wav(&dir.join("speech/a.wav"), 0.10);
        wav(&dir.join("noise/b.wav"), 0.05);

        let subset = [("speech/a.wav", "", 0u64), ("noise/b.wav", "", 0u64)];
        let mut totals = Totals::default();
        feed_files(
            &subset,
            &dir,
            &mut Fixed,
            &mut stream(),
            &vec![0f32; sets::FILE_GAP_SAMPLES],
            Deadline::new(60),
            MemoryCeiling::process_only(4096),
            &mut totals,
        )
        .expect("a synthetic corpus two files long");

        assert_eq!(totals.files_fed, 2);
        assert!(
            (totals.audio_secs - 0.15).abs() < 1e-9,
            "{}",
            totals.audio_secs
        );
        assert!(
            (totals.speech_secs - 0.10).abs() < 1e-9,
            "{}",
            totals.speech_secs
        );
        assert!(
            (totals.noise_secs - 0.05).abs() < 1e-9,
            "{}",
            totals.noise_secs
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Fails the first time it is asked for a window: what a feed that dies
    /// inside a file looks like from the phase's side.
    struct Failing;

    impl Embedder for Failing {
        fn embed_group(&mut self, _windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            anyhow::bail!("the front end failed")
        }
    }

    /// A feed that dies inside a file credits the prefix it consumed, not the
    /// file. This is the defect the phase was fixed for: with the whole clip
    /// credited before the feed, this test reads the file's full length here.
    #[test]
    fn a_feed_that_dies_inside_a_file_credits_the_prefix_it_consumed() {
        let dir =
            std::env::temp_dir().join(format!("mahbot-real-audio-cut-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("speech")).expect("temp corpus");
        wav(&dir.join("speech/a.wav"), 0.5);
        let subset = [("speech/a.wav", "", 0u64)];
        let mut totals = Totals::default();
        let mut stream = stream();
        // One queued window, resolved on the feed's first chunk: the feed then
        // dies with the rest of the file unread.
        stream.queue_window(vec![1.0, 0.0]);
        let died = feed_files(
            &subset,
            &dir,
            &mut Failing,
            &mut stream,
            &vec![0f32; sets::FILE_GAP_SAMPLES],
            Deadline::new(60),
            MemoryCeiling::process_only(4096),
            &mut totals,
        );
        assert!(died.is_err(), "the front end's failure reaches the caller");
        let consumed = stream.samples_fed() as f64 / f64::from(listen::SAMPLE_RATE);
        assert!(consumed > 0.0, "the feed consumed some of the file");
        assert!(consumed < 0.5, "and less than all of it ({consumed} s)");
        assert!(
            (totals.audio_secs - consumed).abs() < 1e-9,
            "credited {} s against {consumed} s consumed",
            totals.audio_secs
        );
        assert_eq!(totals.files_fed, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stop that arrives at a file boundary credits nothing: the run has no
    /// audio, so the report has no rate to state rather than a rate over zero.
    #[test]
    fn a_stop_at_the_first_file_credits_nothing() {
        let dir =
            std::env::temp_dir().join(format!("mahbot-real-audio-stop-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("speech")).expect("temp corpus");
        wav(&dir.join("speech/a.wav"), 0.10);
        let subset = [("speech/a.wav", "", 0u64)];
        let mut totals = Totals::default();
        let stopped = feed_files(
            &subset,
            &dir,
            &mut Fixed,
            &mut stream(),
            &vec![0f32; sets::FILE_GAP_SAMPLES],
            Deadline::new(0),
            MemoryCeiling::process_only(4096),
            &mut totals,
        );
        assert!(stopped.is_err(), "a stop at the first file is an error");
        assert_eq!(totals.files_fed, 0);
        assert_eq!(totals.audio_secs, 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
