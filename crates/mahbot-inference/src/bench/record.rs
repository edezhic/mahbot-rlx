// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bench's two text surfaces — the console rows and the `MEASUREMENTS.md`
//! block — both read out of the report JSON the run itself wrote, so the console
//! and the record cannot state a different number, and a row the report does not
//! carry names its absence instead of printing a zero nobody measured.

use serde_json::Value;

use crate::measurements;

use super::audio::{Category, manifest_count};
use super::rate;
use super::sets;
use super::voices;

/// The per-stream rows, on the console: the voice subset warning, each selected
/// voice's identity, the similarity witness, one line per voice on each side, the
/// pooled figures and the non-speech line — read out of the same report the record
/// is built from, so the two cannot state a different number.
pub(super) fn console(report: &Value) {
    for line in lines(report) {
        eprintln!("{line}");
    }
}

/// [`console`]'s lines, built rather than printed so a test can read them: the
/// console is where a completed run states its pooled figures, and one that omits
/// them is a defect no other surface would show.
pub(super) fn lines(report: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let rows = Rows::read(report);
    let selection = report.get("voices");
    let selected = |key: &str| {
        selection
            .and_then(|node| node.get(key))
            .and_then(Value::as_u64)
    };
    if let Some(voices) = selected("selected")
        && voices + 1 < sets::SPEECH_VOICES as u64
    {
        lines.push(format!(
            "[wake-word] voice subset: {} of the {} voices this measurement is defined over — a \
             screening run, never the record's set",
            voices + 1,
            sets::SPEECH_VOICES
        ));
    }
    for entry in array(selection.and_then(|node| node.get("voices"))) {
        lines.push(format!(
            "[wake-word] voice {}: {} ({})",
            text(entry.get("name")),
            text(entry.get("recording")),
            span_of(entry.get("span"))
        ));
    }
    if let Some(similarity) = selection.and_then(|node| node.get("reference_similarity")) {
        let figure = |key: &str| similarity.get(key).and_then(Value::as_f64);
        if let (Some(min), Some(median), Some(max)) =
            (figure("min"), figure("median"), figure("max"))
        {
            let pair = |key: &str| text(similarity.get("closest").and_then(|node| node.get(key)));
            let cosine = match similarity
                .get("closest")
                .and_then(|closest| closest.get("cosine"))
                .and_then(Value::as_f64)
            {
                Some(cosine) => format!("{cosine:.4}"),
                None => UNAVAILABLE.to_string(),
            };
            lines.push(format!(
                "[wake-word] reference similarity over {} selected voices: min {min:.4}, median \
                 {median:.4}, max {max:.4} · closest pair {} and {} at {cosine}",
                selected("selected").unwrap_or(0),
                pair("a"),
                pair("b")
            ));
        }
    }
    for replacement in array(selection.and_then(|node| node.get("replacements"))) {
        lines.push(format!(
            "[wake-word] replaced: {} ({}) — {}",
            text(replacement.get("recording")),
            span_of(replacement.get("span")),
            text(replacement.get("reason"))
        ));
    }
    for row in &rows.positives {
        lines.push(format!(
            "[wake-word] positives {}: {}",
            row.voice,
            row.cell()
        ));
    }
    lines.push(format!(
        "[wake-word] positives pooled: {}",
        rows.positives_pooled
            .as_ref()
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::line)
    ));
    for row in &rows.negatives {
        lines.push(format!(
            "[wake-word] negatives {}: {} · {}",
            row.voice,
            row.overall,
            row.labelled()
        ));
    }
    lines.push(format!(
        "[wake-word] negatives pooled: {}",
        rows.negatives_pooled
            .as_ref()
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::line)
    ));
    lines.push(match &rows.non_speech {
        Some(cell) => format!("[wake-word] non-speech: {cell} in a stream of its own"),
        None => format!("[wake-word] non-speech: {UNAVAILABLE}"),
    });
    if !rows.non_speech_fired.is_empty() {
        lines.push(format!(
            "[wake-word] non-speech fired: {}",
            rows.non_speech_fired.join(", ")
        ));
    }
    lines
}

/// The negatives' section, `false_reactions.non_phrase_set` — the one place the
/// path is stated.
fn non_phrase(report: &Value) -> Option<&Value> {
    report.get("false_reactions")?.get("non_phrase_set")
}

/// The speech streams, in report order, enrolled first.
fn speech(report: &Value) -> Option<&Value> {
    non_phrase(report)?.get("speech")
}

/// The negatives' own `set` cells: what this run measured, what it falls short of.
fn negative_set(report: &Value) -> Option<&Value> {
    non_phrase(report)?.get("set")
}

/// The 17 non-speech clips, measured in a stream of their own.
fn non_speech(report: &Value) -> Option<&Value> {
    non_phrase(report)?.get("non_speech")
}

/// The positives' `set` cells.
fn positive_set(report: &Value) -> Option<&Value> {
    report.get("recognition")?.get("set")
}

/// The per-voice rows of both sides, read once: the console and the record's tables
/// print the same figures, so neither walks the report a second time.
struct Rows<'a> {
    /// One row per voice on the negative side, enrolled first.
    negatives: Vec<NegativeRow>,
    /// One row per voice on the positive side, enrolled first.
    positives: Vec<PositiveRow>,
    negatives_pooled: Option<Pooled>,
    positives_pooled: Option<Pooled>,
    /// The 17 non-speech clips' cell: one stream, no voice.
    non_speech: Option<String>,
    /// The non-speech clips that fired, as the report names them.
    non_speech_fired: Vec<&'a str>,
    /// The nodes the rows came from: both sides' `set` cells and the speech node, so
    /// a printer that states a caption reads them without walking the report again.
    set: Option<&'a Value>,
    speech: Option<&'a Value>,
    positives_set: Option<&'a Value>,
}

struct NegativeRow {
    voice: String,
    /// The three category cells, in [`Category::SPEECH`] order.
    categories: Vec<String>,
    /// The row's own overall.
    overall: String,
}

impl NegativeRow {
    /// The console's row: the categories labelled, because no table header names
    /// them there.
    fn labelled(&self) -> String {
        Category::SPEECH
            .iter()
            .zip(&self.categories)
            .map(|(category, cell)| format!("{} {cell}", category.label()))
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

struct PositiveRow {
    voice: String,
    /// `(detected, of)`, absent when the report carries no pair for the row.
    counts: Option<(usize, usize)>,
}

impl PositiveRow {
    /// `40/50 (80.0 %)` — the console's figure.
    fn cell(&self) -> String {
        match self.counts {
            Some((detected, of)) => cell_of(detected, of),
            None => UNAVAILABLE.to_string(),
        }
    }

    /// `40 | 50 | 80.0 %` — the record table's three columns.
    fn columns(&self) -> String {
        match self.counts {
            Some((detected, of)) => {
                format!("{detected} | {of} | {}", rate_cell(detected, of))
            }
            None => format!("{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE}"),
        }
    }
}

impl<'a> Rows<'a> {
    fn read(report: &'a Value) -> Self {
        let negatives = array(speech(report).and_then(|node| node.get("voices")))
            .iter()
            .map(|entry| NegativeRow {
                voice: text(entry.get("voice")).to_string(),
                categories: Category::SPEECH
                    .iter()
                    .map(|category| count_cell(entry.get(category.key()), "false_reactions"))
                    .collect(),
                overall: count_cell(Some(entry), "false_reactions"),
            })
            .collect();
        let positives = array(
            report
                .get("recognition")
                .and_then(|node| node.get("voices")),
        )
        .iter()
        .map(|entry| PositiveRow {
            voice: text(entry.get("voice")).to_string(),
            counts: counts(Some(entry), "detected"),
        })
        .collect();
        Self {
            negatives,
            positives,
            negatives_pooled: Pooled::read(
                negative_set(report),
                speech(report),
                "false_reactions",
                "speech_clips_per_voice",
                "trials",
            ),
            positives_pooled: Pooled::read(
                positive_set(report),
                report.get("recognition"),
                "detected",
                "clips_per_voice",
                "utterances",
            ),
            non_speech: counts(non_speech(report), "false_reactions")
                .map(|(fired, of)| cell_of(fired, of)),
            non_speech_fired: array(non_speech(report).and_then(|node| node.get("fired")))
                .iter()
                .filter_map(Value::as_str)
                .collect(),
            set: negative_set(report),
            speech: speech(report),
            positives_set: positive_set(report),
        }
    }

    /// A cell of the negatives' `set`, or [`UNAVAILABLE`].
    fn set_cell(&self, key: &str) -> String {
        count(self.set.and_then(|node| node.get(key)))
    }

    /// A cell of the positives' `set`, or [`UNAVAILABLE`].
    fn positive_cell(&self, key: &str) -> String {
        count(self.positives_set.and_then(|node| node.get(key)))
    }
}

/// One side's overall, with the set it divides: the figure, the per-voice clip
/// count, the distinct texts and the voice count. One rendering, printed by the
/// console and by both tables, so the pooled nature is stated wherever the number is.
struct Pooled {
    fired: usize,
    of: usize,
    clips_per_voice: usize,
    distinct_texts: usize,
    voices: usize,
    unit: &'static str,
}

impl Pooled {
    /// `set` is the side's own cells, `figures` the node carrying the pooled
    /// `fired`/`of`; absent cells mean an absent figure, never a zero.
    fn read(
        set: Option<&Value>,
        figures: Option<&Value>,
        count: &str,
        per_voice: &str,
        unit: &'static str,
    ) -> Option<Self> {
        let (fired, of) = counts(figures, count)?;
        let set = set?;
        let whole = |key: &str| set.get(key)?.as_u64().map(|value| value as usize);
        Some(Self {
            fired,
            of,
            clips_per_voice: whole(per_voice)?,
            distinct_texts: whole("distinct_texts")?,
            voices: whole("voices")?,
            unit,
        })
    }

    /// `440 | 550 | 80.0 %` — the record table's three columns for the pooled row.
    fn columns(&self) -> String {
        format!(
            "{} | {} | {}",
            self.fired,
            self.of,
            rate_cell(self.fired, self.of)
        )
    }

    /// `54/1144 (4.7 %)` — the overall cell of the negatives' table, so the pooled
    /// figure is stated from the one place the console reads it too.
    fn cell(&self) -> String {
        cell_of(self.fired, self.of)
    }

    /// `pooled over 104 clips (56 distinct texts) across 11 voices, not 1144
    /// independent trials` — the clause a table states under its pooled row, where
    /// the figure itself is already in the row above.
    fn clause(&self) -> String {
        format!(
            "pooled over {} clips ({} distinct texts) across {} voices, not {} independent {}",
            self.clips_per_voice, self.distinct_texts, self.voices, self.of, self.unit
        )
    }

    /// `54/1144 (4.7 %) — pooled over …` — the figure with its clause, for the
    /// console and the record's metric row.
    fn line(&self) -> String {
        format!("{} — {}", self.cell(), self.clause())
    }
}

/// The report array under `node`, empty when the report does not carry one.
fn array(node: Option<&Value>) -> &[Value] {
    node.and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// A report string, or [`UNAVAILABLE`].
fn text(node: Option<&Value>) -> &str {
    node.and_then(Value::as_str).unwrap_or(UNAVAILABLE)
}

/// A span cell out of the report: `16.10 s + 15.00 s`, or [`UNAVAILABLE`].
fn span_of(span: Option<&Value>) -> String {
    let number = |key: &str| span.and_then(|span| span.get(key)).and_then(Value::as_f64);
    match (number("offset_secs"), number("secs")) {
        (Some(offset), Some(secs)) => voices::span_label(offset, secs),
        _ => UNAVAILABLE.to_string(),
    }
}

/// What a bench row prints for a figure its report does not carry: the same words
/// every row uses, so no figure is ever printed as a zero the run did not measure.
pub(super) const UNAVAILABLE: &str = "not in the report";

/// One row of the record's table: the figures the row states, or [`UNAVAILABLE`].
fn row(body: &mut String, label: &str, figures: Option<String>) {
    body.push_str(&format!(
        "| {label} | {} |\n",
        figures.as_deref().unwrap_or(UNAVAILABLE)
    ));
}

/// [`row`] for a row of numbers, each bound by name out of the report. The row is
/// written with its render only when every figure it states is present.
macro_rules! figure_row {
    ($body:expr, $label:expr, [$($name:ident = $value:expr),+ $(,)?] => $render:expr) => {
        row(
            &mut $body,
            $label,
            match ($($value.and_then(Value::as_f64),)+) {
                ($(Some($name),)+) => Some($render),
                _ => None,
            },
        )
    };
}

/// The bench section this run's report states — read out of the report the run
/// itself wrote, so the file and the JSON cannot state the same fact twice or a
/// degenerate rate differently.
pub(super) fn block(report: &Value) -> String {
    let mut body = String::new();
    body.push_str("## Wake-word bench\n\n");
    body.push_str(&format!("_Measured {}._\n\n", measurements::now()));
    body.push_str(&format!(
        "status: `{}`\n\n",
        report
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or(UNAVAILABLE)
    ));
    body.push_str("| metric | value |\n|---|---|\n");
    let set = negative_set(report);
    figure_row!(body, "negatives", [
        total = set.and_then(|node| node.get("clips")),
        speech_clips = set.and_then(|node| node.get("speech_clips")),
        non_speech_clips = set.and_then(|node| node.get("non_speech_clips")),
        voices = set.and_then(|node| node.get("voices")),
    ] => format!("{total} clips — {speech_clips} speech across {voices} voices, {non_speech_clips} non-speech in a stream of their own"));
    // Both pooled rows and their tables read the same `Pooled` figures: one
    // rendering of the pooled claim, in three places that cannot disagree.
    let rows = Rows::read(report);
    row(
        &mut body,
        "negative overall (pooled speech)",
        rows.negatives_pooled.as_ref().map(Pooled::line),
    );
    row(
        &mut body,
        "non-speech",
        rows.non_speech
            .as_ref()
            .map(|cell| format!("{cell} — silence and noise, measured apart")),
    );
    row(
        &mut body,
        "positives (pooled)",
        rows.positives_pooled.as_ref().map(Pooled::line),
    );
    let real = report
        .get("false_reactions")
        .and_then(|section| section.get("real_audio"));
    let fa = real.and_then(|section| section.get("fa"));
    let feed = real.and_then(|section| section.get("feed"));
    let hours = |key: &str| feed.and_then(|node| node.get(key)).and_then(Value::as_f64);
    // Nothing is formatted for a row the report cannot state: without an event
    // count the row names its absence.
    match fa
        .and_then(|node| node.get("events"))
        .and_then(Value::as_u64)
    {
        Some(events) => {
            let cells: Vec<String> = [
                (
                    "fa_per_hour_vad_active",
                    "VAD-active speech",
                    "vad_active_hours_fed",
                ),
                ("fa_per_hour_audio", "wall-clock audio", "audio_hours_fed"),
            ]
            .into_iter()
            .map(|(key, label, hours_key)| {
                match fa.and_then(|node| node.get(key)).and_then(Value::as_f64) {
                    Some(rate) => match hours(hours_key) {
                        Some(hours) => format!("{rate:.4}/h over {label} ({hours:.3} h)"),
                        None => format!("{rate:.4}/h over {label}"),
                    },
                    // A rate the report cannot divide names the denominator
                    // that is missing rather than asserting a cause.
                    None if hours(hours_key).is_some_and(|hours| hours <= 0.0) => {
                        format!("no rate: {label} measured no time")
                    }
                    None => format!("no rate: {label} not in the report"),
                }
            })
            .collect();
            let ladder = hours("vad_active_hours_fed")
                .filter(|hours| *hours > 0.0)
                .map(|hours| format!(" · one event = {:.4}/h", 1.0 / hours))
                .unwrap_or_default();
            row(
                &mut body,
                "false accepts",
                Some(format!("{events} events · {}{ladder}", cells.join(" · "))),
            );
        }
        None => row(&mut body, "false accepts", None),
    }
    // The set the run priced, in the vocabulary `--wake-thresholds` reads, so the
    // point can be replayed from the record by copy-paste.
    row(
        &mut body,
        "thresholds",
        report
            .get("thresholds")
            .and_then(Value::as_str)
            .map(|line| format!("`{line}`")),
    );
    let calibration = report.get("calibration");
    let derived = |key: &str| calibration.and_then(|node| node.get(key));
    figure_row!(body, "derived point", [
        floor = derived("derived_soft_floor"),
        fire = derived("fire_factor"),
        bar = derived("static_match_threshold"),
        reset = derived("reset_soft"),
    ] => format!("floor {floor:.4} · fire factor {fire:.4} · static bar {bar:.4} · reset {reset:.4}"));
    figure_row!(body, "wall clock", [
        secs = report.get("wall_clock_secs"),
    ] => format!("{secs:.1} s"));
    figure_row!(body, "peak resident", [
        bytes = report.get("peak_rss_bytes"),
    ] => crate::report::mib(bytes as u64));
    let corpus = report.get("corpus");
    figure_row!(body, "corpus", [
        reused = corpus.and_then(|node| node.get("reused")),
        reads = corpus.and_then(|node| node.get("reads")),
        rendered = corpus.and_then(|node| node.get("rendered")),
        secs = corpus.and_then(|node| node.get("synthesis_secs")),
    ] => format!("{reused} of {reads} clip reads reused, {rendered} rendered in {secs:.1} s"));
    // The build's own account: a warm run renders nothing, so without this row the
    // record would make the corpus look free. `built` is every clip a build of this
    // generation rendered — which exceeds the plan only when the check rejected a
    // candidate, whose probes the plan does not list — so the note explaining the
    // excess is stated only when the excess is there.
    let build = report.get("corpus_build");
    let last = build.and_then(|node| node.get("last"));
    let seconds = |node: Option<&Value>, key: &str| {
        node.and_then(|node| node.get(key)).and_then(Value::as_f64)
    };
    let cells = (
        manifest_count(build, "clips"),
        manifest_count(build, "built"),
        manifest_count(build, "builds"),
        seconds(build, "synthesis_secs"),
        manifest_count(last, "built"),
        seconds(last, "wall_secs"),
    );
    row(
        &mut body,
        "corpus build",
        match cells {
            (Some(clips), Some(built), Some(builds), Some(secs), Some(last_built), Some(wall)) => {
                // "over 7 builds", "over 1 build": a count of builds is a word here.
                let builds = match builds {
                    1 => "1 build".to_string(),
                    builds => format!("{builds} builds"),
                };
                // The note explains an excess that only exists when a candidate was
                // rejected after its probes were rendered, so it appears only then.
                let beyond = if built > clips {
                    " (its renders include the probe renders of candidates the check rejected, \
                     which the plan does not list)"
                } else {
                    ""
                };
                let mut figures = format!(
                    "{clips} clips · {built} rendered over {builds} of this generation{beyond}, \
                     {secs:.0} s of synthesis; the last build took {wall:.0} s wall and rendered \
                     {last_built}"
                );
                match manifest_count(last, "swept") {
                    Some(swept) => {
                        figures.push_str(", clearing ");
                        figures.push_str(&super::audio::superseded(swept));
                    }
                    None => figures.push_str(", and its sweep count is not stated"),
                }
                if let Some(stopped) = last
                    .and_then(|node| node.get("stopped"))
                    .filter(|node| node.is_object())
                {
                    let reason = text(stopped.get("reason"));
                    let outcome = match stopped.get("corpus_complete").and_then(Value::as_bool) {
                        Some(true) => {
                            format!("; that build stopped after the corpus was complete ({reason})")
                        }
                        _ => format!(
                            "; that build stopped before the corpus was complete ({reason}), and \
                             its renders are counted above"
                        ),
                    };
                    figures.push_str(&outcome);
                }
                Some(figures)
            }
            _ => None,
        },
    );
    figure_row!(body, "windows per encoder pass", [
        most = report.get("windows_per_pass"),
        capacity = report.get("group_capacity"),
        partial = report.get("partial_passes"),
        passes = report.get("encoder_passes"),
    ] => format!("at most {most} of {capacity} · {partial} partial of {passes} passes"));
    body.push('\n');
    push_voice_tables(&mut body, &rows);
    push_voice_prose(&mut body, report, &rows);
    body.push_str("| phase | seconds |\n|---|---|\n");
    if let Some(phases) = report.get("phases").and_then(Value::as_object) {
        for (name, secs) in phases {
            row(
                &mut body,
                name,
                secs.as_f64().map(|secs| format!("{secs:.1}")),
            );
        }
    }
    body.push_str("\nProvenance:\n\n");
    body.push_str(&format!("- machine: {}\n", measurements::machine()));
    body.push_str(&format!("- device: {}\n", measurements::device()));
    body.push_str(&format!("- profile: {}\n", measurements::profile()));
    let guards = report.get("guards");
    let bound = |key: &str| {
        guards
            .and_then(|node| node.get(key))
            .and_then(Value::as_u64)
    };
    if let (Some(stop), Some(resident), Some(swap)) = (
        bound("timeout_secs"),
        bound("max_rss_mib"),
        bound("swap_growth_mib"),
    ) {
        body.push_str(&format!(
            "- guards: stop {stop} s · resident {resident} MiB · swap growth {swap} MiB\n"
        ));
    }
    body.push_str(&format!("- code: {}\n", measurements::code_state()));
    body.push_str(&format!(
        "- command: `{}`\n",
        report
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or(UNAVAILABLE)
    ));
    let models = report.get("models");
    let pin = |section: &str, key: Option<&str>| -> Option<&Value> {
        let node = models?.get(section)?;
        key.map_or(Some(node), |key| node.get(key))
    };
    for (role, value) in [
        ("ASR body", pin("recognition", Some("body"))),
        ("ASR sound", pin("recognition", Some("sound"))),
        ("TTS body", pin("synthesis", Some("body"))),
        ("TTS sound", pin("synthesis", Some("sound"))),
        ("voice", pin("voice_wav", None)),
    ] {
        // One behaviour for an absent pin: the bullet is written and names the
        // absence, whether the report has no pin or a pin without the figure.
        let Some(pin) = value else {
            body.push_str(&format!("- {role}: {UNAVAILABLE}\n"));
            continue;
        };
        let bytes = pin
            .get("bytes")
            .and_then(Value::as_u64)
            .map(|bytes| format!("{bytes} bytes"))
            .unwrap_or_else(|| UNAVAILABLE.to_string());
        body.push_str(&format!(
            "- {role}: `{}` · {bytes} · sha256 {}\n",
            pin.get("file")
                .and_then(Value::as_str)
                .unwrap_or(UNAVAILABLE),
            pin.get("sha256")
                .and_then(Value::as_str)
                .unwrap_or(UNAVAILABLE)
        ));
    }
    if let Some(entries) = report
        .get("voices")
        .and_then(|node| node.get("voices"))
        .and_then(Value::as_array)
    {
        for entry in entries {
            body.push_str(&format!(
                "- {}: `{}` · {} · sha256 {}\n",
                text(entry.get("name")),
                text(entry.get("recording")),
                span_of(entry.get("span")),
                text(entry.get("sha256"))
            ));
        }
    }
    body
}

/// The record's per-voice tables: one row per stream on each side, the pooled row,
/// and the non-speech line. Captions, denominators and distinct-text counts are read
/// out of each side's own `set` cell, so a phrase-list edit moves them without an
/// edit here; each voice's recording and span are stated once, in the provenance
/// bullets above. The rows themselves are the [`Rows`] the console prints, so the
/// report is walked once for both surfaces.
fn push_voice_tables(body: &mut String, rows: &Rows<'_>) {
    let mut header = String::from("| voice |");
    let mut rule = String::from("|---|");
    for category in Category::SPEECH {
        header.push_str(&format!(
            " {} ({}) |",
            category.label(),
            rows.set_cell(&format!("{}_clips_per_voice", category.key()))
        ));
        rule.push_str("---|");
    }
    header.push_str(&format!(
        " overall ({}) |\n",
        rows.set_cell("speech_clips_per_voice")
    ));
    rule.push_str("---|\n");
    body.push_str(&format!(
        "Negatives by voice — {} speech clips each:\n\n{header}{rule}",
        rows.set_cell("speech_clips_per_voice")
    ));
    for row in &rows.negatives {
        body.push_str(&format!(
            "| {} | {} | {} |\n",
            row.voice,
            row.categories.join(" | "),
            row.overall
        ));
    }
    let mut line = format!("| pooled over {} voices |", rows.set_cell("voices"));
    for category in Category::SPEECH {
        line.push_str(&format!(
            " {} |",
            count_cell(
                rows.speech.and_then(|node| node.get(category.key())),
                "false_reactions"
            )
        ));
    }
    line.push_str(&format!(
        " {} |\n\n",
        rows.negatives_pooled
            .as_ref()
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::cell)
    ));
    body.push_str(&line);
    body.push_str(&format!(
        "{}\n\n",
        rows.negatives_pooled
            .as_ref()
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::clause)
    ));
    body.push_str(&format!(
        "Non-speech, in a stream of its own: {}.\n\n",
        rows.non_speech.as_deref().unwrap_or(UNAVAILABLE)
    ));

    body.push_str(&format!(
        "Positives by voice — {} clips each:\n\n\
         | voice | detected | of | rate |\n|---|---|---|---|\n",
        rows.positive_cell("clips_per_voice")
    ));
    for row in &rows.positives {
        body.push_str(&format!("| {} | {} |\n", row.voice, row.columns()));
    }
    let pooled = rows.positives_pooled.as_ref().map_or_else(
        || format!("{UNAVAILABLE} | {UNAVAILABLE} | {UNAVAILABLE}"),
        Pooled::columns,
    );
    body.push_str(&format!(
        "| pooled over {} voices | {pooled} |\n\n{}\n\n",
        rows.positive_cell("voices"),
        rows.positives_pooled
            .as_ref()
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::clause)
    ));
}

/// A report count as text, or [`UNAVAILABLE`].
fn count(node: Option<&Value>) -> String {
    node.and_then(Value::as_u64)
        .map_or_else(|| UNAVAILABLE.to_string(), |figure| figure.to_string())
}

/// The `(fired, of)` a report cell carries under `key`, or `None` when the report
/// does not have both figures.
fn counts(node: Option<&Value>, key: &str) -> Option<(usize, usize)> {
    let fired = node
        .and_then(|node| node.get(key))
        .and_then(Value::as_u64)?;
    let of = node
        .and_then(|node| node.get("of"))
        .and_then(Value::as_u64)?;
    Some((usize::try_from(fired).ok()?, usize::try_from(of).ok()?))
}

/// `fired/of (rate %)` — the cell a row states, with a degenerate denominator named
/// rather than divided by.
fn cell_of(fired: usize, of: usize) -> String {
    format!("{fired}/{of} ({})", rate_cell(fired, of))
}

/// The rate a cell states: `80.0 %`, or `no rate` where the report carries no clips
/// to divide — a figure that is present, unlike [`UNAVAILABLE`].
fn rate_cell(fired: usize, of: usize) -> String {
    if of == 0 {
        return "no rate".to_string();
    }
    format!("{:.1} %", 100.0 * rate(fired, of))
}

/// One report cell as `fired/of (rate %)`, or [`UNAVAILABLE`] when the report does
/// not carry it.
fn count_cell(node: Option<&Value>, key: &str) -> String {
    counts(node, key).map_or_else(|| UNAVAILABLE.to_string(), |(fired, of)| cell_of(fired, of))
}

/// The record's prose: what the numbers are, what they are not, and the limits
/// that make them readable.
fn push_voice_prose(body: &mut String, report: &Value, rows: &Rows<'_>) {
    let voices = rows.set_cell("voices");
    let clones = report
        .get("voices")
        .and_then(|node| node.get("selected"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    body.push_str(&format!(
        "**What these numbers are.** The prototype, the calibration pool and the \
        anti-prototypes are the enrolled voice's alone; the {clones} clone voices are scored \
        against them, which is what makes their rows generalisation evidence. The enrolled \
        voice's row is \
        measured at thresholds fitted on that same voice and is not.\n\n"
    ));
    body.push_str(
        "**The voices.** The clone voices are rendered from as many distinct single-reader \
        recordings in the corpus input that the false-accept phase does not measure, one per \
        voice, each named in the provenance block below with the recording it was cut from and \
        its span. A clone's only per-voice input is a speaker embedding, so their renders may \
        resemble one another more than their sources do: these figures describe how the detector \
        behaves across these voice variations, not how it would behave for arbitrary strangers. \
        The corpus establishes single-reader provenance, not speaker identity: these are distinct \
        recordings, not verified different people.\n\n",
    );
    let similarity = report
        .get("voices")
        .and_then(|node| node.get("reference_similarity"));
    let figure = |key: &str| {
        similarity
            .and_then(|node| node.get(key))
            .and_then(Value::as_f64)
    };
    if let (Some(min), Some(median), Some(max)) = (figure("min"), figure("median"), figure("max")) {
        let closest = similarity
            .and_then(|node| node.get("closest"))
            .map(
                |closest| match closest.get("cosine").and_then(Value::as_f64) {
                    Some(cosine) => format!(
                        "{} and {} at {cosine:.4}",
                        text(closest.get("a")),
                        text(closest.get("b"))
                    ),
                    None => UNAVAILABLE.to_string(),
                },
            )
            .unwrap_or_else(|| UNAVAILABLE.to_string());
        let pairs = similarity
            .and_then(|node| node.get("pairs"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        body.push_str(&format!(
            "The witness that the clone voices did not collapse into one voice: the pairwise cosine \
             between the reference embeddings the clones are built from is min {min:.4}, median \
             {median:.4}, max {max:.4} over {pairs} pairs, the closest being {closest}. Nothing was \
             rejected on it.\n\n"
        ));
    }
    // Named, not positional: a reordering that silently states the wrong number is
    // the one mistake a `format!` this wide cannot catch.
    let negatives = rows.set_cell("clips");
    let speech_clips = rows.set_cell("speech_clips");
    let per_voice = rows.set_cell("speech_clips_per_voice");
    let near_misses = rows.set_cell("confusable_clips_per_voice");
    let unrelated = rows.set_cell("unrelated_clips_per_voice");
    let in_passing = rows.set_cell("phrase_in_passing_clips_per_voice");
    let non_speech_clips = rows.set_cell("non_speech_clips");
    let positives = rows.positive_cell("clips_per_voice");
    let wake_only = rows.positive_cell("wake_only_clips_per_voice");
    let with_command = rows.positive_cell("command_plus_speech_clips_per_voice");
    let positive_stream = count(report.get("recognition").and_then(|node| node.get("of")));
    let negative_stream = count(rows.speech.and_then(|node| node.get("of")));
    body.push_str(&format!(
        "**The set.** The negative side is {negatives} clips — {speech_clips} speech clips \
         ({per_voice} per voice: {near_misses} phonetic near-misses, {unrelated} unrelated \
         sentences, {in_passing} wake-phrase-in-passing clips, over {voices} voices) and the \
         {non_speech_clips} non-speech clips measured apart in a stream of their own — and the \
         positive side {positive_stream} clips ({positives} per voice: {wake_only} bare \
         wake-phrase renders plus {with_command} wake-phrase-plus-command clips, the same \
         {wake_only} and {with_command} in every voice). Every pooled figure is pooled: the same \
         {per_voice} (or {positives}) clips across {voices} voices, not {negative_stream} (or \
         {positive_stream}) independent utterances. The overalls over the {clones} clone voices \
         are not comparable with the enrolled row and are not a regression against anything — a \
         different population, and thresholds fitted on another voice. A near-zero figure on \
         either side is a finding about cross-voice detection, not a failed round: the detector \
         was not re-tuned here.\n\n"
    ));
    let enrolled = report.get("voices").and_then(|node| node.get("enrolled"));
    let enrolled_outcome = match enrolled
        .and_then(|node| node.get("passed"))
        .and_then(Value::as_bool)
    {
        Some(true) => "passed",
        Some(false) => {
            "failed, and the run completed and recorded anyway — a fact about the owner's \
                        clip, not grounds to stop a round, and never a reason to replace it"
        }
        None => UNAVAILABLE,
    };
    // The probes the report lists, in full: the check scores every sentence of the
    // pinned list, so the record states that list rather than a copy of it here.
    let probes: Vec<String> = array(
        report
            .get("voices")
            .and_then(|node| node.get("intelligibility"))
            .and_then(|node| node.get("sentences")),
    )
    .iter()
    .map(|sentence| format!("`{}`", text(Some(sentence))))
    .collect();
    body.push_str(&format!(
        "**The check.** A voice counts as usable when the pinned ordinary sentences — {} — \
        rendered in it come back from the workspace's own recogniser word for word, in order, \
        ignoring case and punctuation. The enrolled voice's own outcome: {enrolled_outcome}. The \
        check is deliberately not built on the wake phrase, which this recogniser spells in no \
        voice, including the owner's; whether the wake phrase survives in a given voice is what \
        that voice's rows above measure. The check establishes that a voice is usable; the rest of \
        its renders are not individually gated.\n\n",
        if probes.is_empty() {
            UNAVAILABLE.to_string()
        } else {
            probes.join(", ")
        },
    ));
    let selection = report.get("voices");
    let selected = selection
        .and_then(|node| node.get("selected"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let pool = selection
        .and_then(|node| node.get("pool"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut selection_note = format!(
        "The voices were selected by walking the pinned pool of {pool} recordings in its \
         pinned order and taking the first that passed the check."
    );
    if let Some(replacements) = selection
        .and_then(|node| node.get("replacements"))
        .and_then(Value::as_array)
        && !replacements.is_empty()
    {
        let named: Vec<String> = replacements
            .iter()
            .map(|replacement| {
                format!(
                    "`{}` ({}) — {}",
                    text(replacement.get("recording")),
                    span_of(replacement.get("span")),
                    text(replacement.get("reason"))
                )
            })
            .collect();
        selection_note.push_str(&format!(
            " {} rejected and replaced after failing the check, in the order tried: {}.",
            named.len(),
            named.join("; ")
        ));
    }
    body.push_str(&selection_note);
    body.push_str(&format!(
        " The measured set is {selected} clone voices plus the enrolled voice.\n\n"
    ));
}
