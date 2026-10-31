// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bench's two text surfaces — the console rows and the `WAKE-WORD-BENCH.md`
//! tables — both read out of the report JSON the run itself wrote, so the console
//! and the record cannot state a different number, and a row the report does not
//! carry names its absence instead of printing a zero nobody measured.

use serde_json::Value;

use super::audio::Category;
use super::config::BenchConfig;
use super::rate;
use super::sets;
use super::voices;
use crate::measurements;

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

    /// `Pooled over 11 voices: 104 clips each, not 1144 independent trials.` — the
    /// one line a table states under its pooled row, where the figure itself is
    /// already in the row above. It opens with the pooled row's own words, so it
    /// reads as that row's caveat rather than as a title for the next table, and it
    /// states the per-voice count once, because the row above already divides by it.
    fn note(&self) -> String {
        format!(
            "Pooled over {} voices: {} clips each, not {} independent {}.",
            self.voices, self.clips_per_voice, self.of, self.unit
        )
    }

    /// `pooled over 104 clips (56 distinct texts) across 11 voices, not 1144
    /// independent trials` — the clause the metric row states after its figure.
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

/// The bench's tables this run's report states — read out of the report the run
/// itself wrote, so the file and the JSON cannot state the same fact twice or a
/// degenerate rate differently.
pub(super) fn block(report: &Value) -> String {
    let mut body = String::new();
    body.push_str("| metric | value |\n|---|---|\n");
    row(
        &mut body,
        "corpus",
        report
            .get("corpus")
            .and_then(|node| node.get("clips"))
            .and_then(Value::as_u64)
            .map(|clips| format!("{clips} clips")),
    );
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
        rows.negatives_pooled.as_ref().map(Pooled::cell),
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
        rows.positives_pooled.as_ref().map(Pooled::cell),
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
    ] => crate::report::human(bytes as u64));
    body.push_str(
        "\nOne complete run: recall and false reactions in each of the eleven voices, and the \
         false accepts per hour over the pinned real-audio corpus.\n\n",
    );
    push_voice_tables(&mut body, &rows);
    body
}

/// The bench's inputs, as the provenance block's bullets: where the run read them
/// from — the locations the stand and corpus records name too, so a reader can
/// point a re-run at the same material — then every model file and every
/// verification voice, with its size and SHA-256.
pub(super) fn record_inputs(cfg: &BenchConfig, report: &Value) -> String {
    let mut body = String::new();
    match &cfg.models {
        Some(dir) => body.push_str(&format!("- models: `{}`\n", measurements::short_path(dir))),
        None => body.push_str(
            "- models: per-role files, named by `--asr-body`, `--asr-sound`, `--tts-body` and \
             `--tts-sound`\n",
        ),
    }
    body.push_str(&format!(
        "- corpus: `{}`\n",
        measurements::short_path(&cfg.corpus)
    ));
    let models = report.get("models");
    let pin = |section: &str, key: Option<&str>| -> Option<&Value> {
        let node = models?.get(section)?;
        key.map_or(Some(node), |key| node.get(key))
    };
    // The voice's bullet states its location — it is the one input that lives
    // outside the model directory — and every other file its name there.
    let voice = measurements::short_path(&cfg.voice);
    for (role, location, value) in [
        ("ASR body", None, pin("recognition", Some("body"))),
        ("ASR sound", None, pin("recognition", Some("sound"))),
        ("TTS body", None, pin("synthesis", Some("body"))),
        ("TTS sound", None, pin("synthesis", Some("sound"))),
        ("voice", Some(voice.as_str()), pin("voice_wav", None)),
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
        let file = location.map_or_else(
            || {
                pin.get("file")
                    .and_then(Value::as_str)
                    .unwrap_or(UNAVAILABLE)
            },
            |path| path,
        );
        body.push_str(&format!(
            "- {role}: `{file}` · {bytes} · sha256 {}\n",
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

/// The record's per-voice tables: one row per stream on each side and the pooled
/// row, each table followed by its caption — the pooled row's own caveat, built by
/// [`Pooled::note`], never a heading that also repeats a count. Captions,
/// denominators and distinct-text counts are read out of each side's own `set` cell,
/// so a phrase-list edit moves them without an edit here; each voice's recording and
/// span are stated once, in the provenance bullets above. The rows themselves are the
/// [`Rows`] the console prints, so the report is walked once for both surfaces.
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
    body.push_str(&header);
    body.push_str(&rule);
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
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::note)
    ));

    body.push_str("| voice | detected | of | rate |\n|---|---|---|---|\n");
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
            .map_or_else(|| UNAVAILABLE.to_string(), Pooled::note)
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
