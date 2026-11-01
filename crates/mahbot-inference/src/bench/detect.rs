// RLX — versatile ML compiler & runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One detection stream: the live wake-word loop with the global state
//! removed, so a stream carries nothing but its own audio.
//!
//! A stream is fed in arbitrary slices and walks them in 512-sample chunks,
//! exactly as a live microphone delivers them. It keeps:
//!
//! * a capped raw ring and a VAD cursor — the VAD loop walks the unconsumed
//!   tail, so the overlap between chunks survives a call boundary;
//! * the VAD-gated speech window, capped at [`listen::WINDOW_SAMPLES`];
//! * the rolling score window and the adaptive threshold;
//! * the segment counters (19 consecutive VAD-negative hops end a segment).
//!
//! Windows are queued and embedded in groups — [`listen::PAIR`] for the
//! product's form, [`listen::GROUP`] on the offline sweep where that pays, one at
//! a time everywhere else. A grouped stream applies its decisions in the order the
//! windows arrived, so nothing is reordered; a group that contains a fire discards
//! the windows after it, because the stream would not have scored them; and the
//! cooldown is measured from the fire's own position in the stream, not from the
//! moment the group was resolved.
//!
//! [`feed`](DetectionStream::feed) never scores a group that is not full, and
//! [`flush`](DetectionStream::flush) is what scores it. Each caller flushes at its own
//! boundary: the offline phases at the end of a clip's turn and of a file's audio and
//! its silence gap — the boundaries the sweep's figures were taken with, so no group
//! spans a feed — and the product's paired stream
//! ([`with_group`](DetectionStream::with_group) at [`listen::PAIR`]) at the end of the
//! caller's audio, paying one stride of delay for it: a fire resolves at the feed that
//! brought its neighbour, never later.
//!
//! Two rules keep a grouped stream faithful to the ungrouped one:
//!
//! * a segment close — which clears the rolling score window and the adaptive
//!   state — never overtakes queued windows: everything pending is applied
//!   first, so a window is always scored in the state it was produced in;
//! * the recall and false-reaction sets run ungrouped ([`DetectionStream::warm`]),
//!   so their decisions do not depend on a group boundary at all.
//!
//! One deviation remains on the grouped path and is bounded rather than hidden:
//! by the time a group is resolved the detector has already run the hops of the
//! audio that arrived while it was filling, so after a fire it has seen up to one
//! group of audio the ungrouped loop would have held back during the cooldown, and
//! the VAD-active clock counts those hops — 2.56 s for the sweep's sixteen, one
//! stride for the product's two. The sweep's denominators and its rate carry that
//! statement; the recall and false-reaction figures do not, since those phases are
//! ungrouped.
//!
//! Two behaviours are specific to the bench:
//!
//! * the **cooldown is an audio clock** — a wall clock at the bench's
//!   faster-than-realtime feed would suppress far more audio than the 3 s it
//!   represents; here 3 s of fed audio re-arms the stream;
//! * a **fire is attributed by position, not by the caller's clock**: the stream
//!   accumulates the positions where the fires it accepts were scored, and the caller
//!   takes the fires that landed inside its own unit of audio
//!   ([`take_fires`](DetectionStream::take_fires)).
//!
//! The detector instance, the adaptive state and the cooldown clock survive
//! every reset; only the per-segment state is cleared. [`feed`](DetectionStream::feed)
//! checks the run's wall-clock stop on every chunk, so the long detection phases
//! stop where the time is spent.

use anyhow::Result;

use crate::listen::{self, Embedder, SCORE_STRIDE_SAMPLES};

use crate::wake::{self, AdaptiveThreshold, DetectionGeometry, Enrollment};

use crate::guard::Deadline;

/// The detector's own frame, hop and threshold live in [`crate::vad`], which the
/// transcription path decides on as well: one implementation, two callers.
pub(crate) use crate::vad::{FRAME_LENGTH, HOP_LENGTH, is_speech, vad_gate};
/// Consecutive VAD-negative hops that end a speech segment.
pub(crate) const SEGMENT_TIMEOUT_HOPS: usize = 19;
/// Re-arm delay after an accepted fire, in samples at the listening rate: 3 s
/// of fed audio, not wall-clock time.
pub(crate) const COOLDOWN_SAMPLES: u64 = 3 * listen::SAMPLE_RATE as u64;
/// Raw-ring cap: the window plus enough slack to cover a scoring stride — a
/// score reads a whole window plus one stride of hops out of the ring, so the
/// cap has to hold both or a score sees samples the next feed overwrote.
const AUDIO_BUFFER_MAX: usize = listen::WINDOW_SAMPLES + 8 * SCORE_STRIDE_SAMPLES;

/// One continuous-listening stream.
pub(crate) struct DetectionStream {
    enrollment: Enrollment,
    geometry: DetectionGeometry,
    detector: earshot::Detector,
    audio_buffer: Vec<f32>,
    vad_cursor: usize,
    speech_window: Vec<f32>,
    score_window: Vec<f32>,
    adaptive: AdaptiveThreshold,
    segment_silence_hops: usize,
    segment_score_steps: usize,
    last_score_sample_count: usize,
    samples_fed: u64,
    cooldown_left: u64,
    vad_speech_hops: u64,
    /// Where the fires this stream accepted were scored, in samples fed: the
    /// stream owns them so a stop in the middle of a slice still reports what it
    /// fired on, and so a paired stream's decision can be compared with an
    /// ungrouped one's at the position it was made.
    fires: Vec<u64>,
    /// Windows waiting for the group's encoder pass, with the stream position
    /// each was scored at.
    pending: Vec<Vec<f32>>,
    pending_at: Vec<u64>,
    /// Windows per resolve: [`listen::PAIR`] for the product's two-window pass,
    /// [`listen::GROUP`] where the offline sweep's grouping pays, one where the
    /// set is small enough that the ungrouped order is worth more than the pass.
    group: usize,
}

impl DetectionStream {
    /// A warm, ungrouped stream: the adaptive threshold starts past its
    /// bootstrap, seeded with this enrolment's own static threshold as the
    /// harbour, the model's graphs are already compiled, and every window is
    /// embedded and applied on its own — the recall and false-reaction sets.
    pub(crate) fn warm(enrollment: Enrollment) -> Self {
        Self::with_group(enrollment, 1)
    }

    /// The same stream with its windows embedded in groups of [`listen::GROUP`]:
    /// the real-audio sweep, where one encoder pass per sixteen windows took the
    /// phase from 693.6 s on two ungrouped streams to ≈296 s on one.
    pub(crate) fn grouped(enrollment: Enrollment) -> Self {
        Self::with_group(enrollment, listen::GROUP)
    }

    fn with_group(enrollment: Enrollment, group: usize) -> Self {
        let geometry = enrollment.geometry();
        let adaptive = AdaptiveThreshold::warmed(&enrollment.thresholds, geometry.static_threshold);
        Self {
            enrollment,
            geometry,
            detector: earshot::Detector::default(),
            audio_buffer: Vec::new(),
            vad_cursor: 0,
            speech_window: Vec::new(),
            score_window: Vec::new(),
            adaptive,
            segment_silence_hops: 0,
            segment_score_steps: 0,
            last_score_sample_count: 0,
            samples_fed: 0,
            cooldown_left: 0,
            vad_speech_hops: 0,
            fires: Vec::new(),
            pending: Vec::new(),
            pending_at: Vec::new(),
            group: group.max(1),
        }
    }

    /// Windows per resolve — `1` for the ungrouped streams the clip sets are
    /// measured on. Read by the test that pins which constructor each phase takes.
    #[cfg(test)]
    pub(crate) fn group(&self) -> usize {
        self.group
    }

    /// Queue one window as a scored speech hop would, so a test can reach the
    /// group's resolve without a VAD decision to lean on.
    #[cfg(test)]
    pub(crate) fn queue_window(&mut self, window: Vec<f32>) {
        self.pending.push(window);
        self.pending_at.push(self.samples_fed);
    }

    /// Samples this stream has consumed, gap included. Read around a feed call to
    /// credit a file with the audio that feed actually took, not the whole file.
    pub(crate) fn samples_fed(&self) -> u64 {
        self.samples_fed
    }

    /// VAD-positive hops since the last read, for the VAD-active denominator.
    pub(crate) fn take_vad_hops(&mut self) -> u64 {
        core::mem::take(&mut self.vad_speech_hops)
    }

    /// Fires accepted since the last read.
    pub(crate) fn take_fires(&mut self) -> usize {
        let fires = self.fires.len();
        self.fires.clear();
        fires
    }

    /// Where those fires were scored, for the tests that hold a paired stream to
    /// an ungrouped one's decisions.
    #[cfg(test)]
    pub(crate) fn take_fire_positions(&mut self) -> Vec<u64> {
        core::mem::take(&mut self.fires)
    }

    /// Whether the current speech segment is still open — fewer than
    /// [`SEGMENT_TIMEOUT_HOPS`] consecutive VAD-negative hops since the last
    /// speech. Once it closes, no further fire is possible until new speech.
    pub(crate) fn segment_open(&self) -> bool {
        self.segment_silence_hops < SEGMENT_TIMEOUT_HOPS
    }

    /// Feed one slice. The run's wall-clock stop is checked on every 512-sample
    /// chunk, so a stop cannot be delayed by the whole slice; the fires it
    /// accepted are read with [`take_fires`](Self::take_fires), so a stop in the
    /// middle of a slice still leaves the fires that slice produced. A window
    /// still waiting for its group when the slice ends is left for
    /// [`flush`](Self::flush), which is where a stream scores its tail.
    pub(crate) fn feed<E: Embedder>(
        &mut self,
        samples: &[f32],
        embed: &mut E,
        deadline: Deadline,
        phase: &'static str,
    ) -> Result<()> {
        for chunk in samples.chunks(FRAME_LENGTH) {
            deadline.check(phase)?;
            self.samples_fed += chunk.len() as u64;
            if self.cooldown_left > 0 {
                self.cooldown_left = self.cooldown_left.saturating_sub(chunk.len() as u64);
                self.push(chunk);
                continue;
            }
            self.push(chunk);
            let speech_seen = self.run_hops();
            // A segment close clears the rolling scores and the adaptive state;
            // windows already queued were scored before it in the ungrouped loop,
            // so they are applied here first and never see the reset.
            if self.segment_silence_hops >= SEGMENT_TIMEOUT_HOPS && !self.pending.is_empty() {
                self.resolve(embed)?;
            }
            self.close_segment_if_silent();
            self.score(speech_seen)?;
            if self.pending.len() >= self.group {
                self.resolve(embed)?;
            }
        }
        Ok(())
    }

    /// Score the windows a slice left waiting — the end of the stream, or the end
    /// of one caller's unit of audio. A lone window is embedded on its own: the
    /// paired form's pass is two windows, and padding one into it would cost the
    /// pair's work for a window that has no neighbour.
    pub(crate) fn flush<E: Embedder>(
        &mut self,
        embed: &mut E,
        deadline: Deadline,
        phase: &'static str,
    ) -> Result<()> {
        deadline.check(phase)?;
        self.resolve(embed)
    }

    fn push(&mut self, chunk: &[f32]) {
        let excess = push_capped(&mut self.audio_buffer, chunk, AUDIO_BUFFER_MAX);
        self.vad_cursor = self.vad_cursor.saturating_sub(excess);
    }

    /// Walk the VAD hops over the unconsumed ring. Returns whether any hop in
    /// this chunk was speech.
    fn run_hops(&mut self) -> bool {
        let mut speech_seen = false;
        let mut hop_count = self.segment_silence_hops;
        while self.vad_cursor + FRAME_LENGTH <= self.audio_buffer.len() {
            let frame = &self.audio_buffer[self.vad_cursor..self.vad_cursor + FRAME_LENGTH];
            let hop = &frame[..HOP_LENGTH];
            if is_speech(&mut self.detector, hop) {
                speech_seen = true;
                hop_count = 0;
                self.vad_speech_hops += 1;
                push_capped(&mut self.speech_window, hop, listen::WINDOW_SAMPLES);
            } else {
                hop_count += 1;
            }
            self.vad_cursor += HOP_LENGTH;
            self.last_score_sample_count += HOP_LENGTH;
        }
        self.segment_silence_hops = hop_count;
        speech_seen
    }

    /// A segment ends after [`SEGMENT_TIMEOUT_HOPS`] consecutive VAD-negative
    /// hops: its rolling scores, adaptive state and segment clock are cleared
    /// so nothing accumulates across utterances.
    fn close_segment_if_silent(&mut self) {
        if self.segment_silence_hops < SEGMENT_TIMEOUT_HOPS {
            return;
        }
        self.score_window.clear();
        self.adaptive.reset();
        self.segment_silence_hops = 0;
        self.segment_score_steps = 0;
        self.audio_buffer.clear();
        self.speech_window.clear();
        self.vad_cursor = 0;
        self.last_score_sample_count = 0;
    }

    /// One scoring step when the stride has elapsed and there is something to
    /// score: the window joins the group waiting for the next encoder pass.
    fn score(&mut self, speech_seen: bool) -> Result<()> {
        if self.last_score_sample_count < SCORE_STRIDE_SAMPLES {
            return Ok(());
        }
        if !speech_seen && self.score_window.is_empty() {
            return Ok(());
        }
        self.last_score_sample_count = 0;

        let start = self
            .speech_window
            .len()
            .saturating_sub(listen::WINDOW_SAMPLES);
        if start == self.speech_window.len() {
            return Ok(());
        }
        self.pending.push(self.speech_window[start..].to_vec());
        self.pending_at.push(self.samples_fed);
        Ok(())
    }

    /// Embed the windows waiting for a full group, or for a flush, and apply their
    /// decisions in arrival order.
    fn resolve<E: Embedder>(&mut self, embed: &mut E) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let windows = std::mem::take(&mut self.pending);
        let at = std::mem::take(&mut self.pending_at);
        let embeddings = embed.embed_group(&windows)?;
        self.apply(&embeddings, &at);
        Ok(())
    }

    /// Apply a group's scores in arrival order. A fire ends the walk: the
    /// remaining windows of the group are dropped, exactly as the stream would
    /// not have scored them, and the cooldown starts at the fire's own position
    /// in the audio, not at the moment the group was resolved.
    /// Returns where the group fired, if it did.
    fn apply(&mut self, embeddings: &[Vec<f32>], at: &[u64]) -> Option<u64> {
        let thresholds = self.enrollment.thresholds;
        for (embedding, &fired_at) in embeddings.iter().zip(at) {
            let total = self.enrollment.soft_score(embedding);
            let adaptive_value = if total < self.geometry.reset {
                self.adaptive
                    .feed(total, &thresholds, self.geometry.static_threshold)
            } else {
                self.adaptive
                    .peek(&thresholds, self.geometry.static_threshold)
            };
            let threshold = adaptive_value.unwrap_or(self.geometry.static_threshold);
            let (fired, _) =
                wake::process_score(total, &mut self.score_window, self.geometry, threshold);

            self.segment_score_steps += 1;
            if !fired || self.segment_score_steps > wake::WAKE_WORD_MAX_SEGMENT_STEPS {
                continue;
            }

            let since = self.samples_fed.saturating_sub(fired_at);
            self.cooldown_left = COOLDOWN_SAMPLES.saturating_sub(since);
            self.fires.push(fired_at);
            self.soft_reset();
            return Some(fired_at);
        }
        None
    }

    /// The post-fire reset: the ring, the speech window and the rolling scores
    /// go; the detector, the adaptive state and the cooldown clock stay.
    fn soft_reset(&mut self) {
        self.audio_buffer.clear();
        self.vad_cursor = 0;
        self.speech_window.clear();
        self.score_window.clear();
        self.segment_silence_hops = 0;
        self.segment_score_steps = 0;
        self.last_score_sample_count = 0;
    }
}

/// Append, then drain the front past `cap`. Returns the number drained so a
/// cursor into the buffer can be shifted by the same amount.
fn push_capped(buf: &mut Vec<f32>, samples: &[f32], cap: usize) -> usize {
    buf.extend_from_slice(samples);
    let excess = buf.len().saturating_sub(cap);
    buf.drain(..excess);
    excess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_keeps_its_tail_and_reports_the_shift() {
        let mut buf = vec![0.0f32; 10];
        let drained = push_capped(&mut buf, &[1.0, 2.0, 3.0], 8);
        assert_eq!(drained, 5);
        assert_eq!(buf, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(push_capped(&mut buf, &[4.0], 8), 1);
    }

    /// A full window read at one scoring step must still be in the ring after
    /// the next chunk arrives: the cap holds the window plus the stride the step
    /// consumes, so the push that follows cannot have drained any of it.
    #[test]
    fn a_scoring_window_survives_the_push_that_follows_it() {
        let first: Vec<f32> = (0..AUDIO_BUFFER_MAX).map(|i| i as f32).collect();
        let mut buffer = Vec::new();
        push_capped(&mut buffer, &first, AUDIO_BUFFER_MAX);
        let drained = push_capped(&mut buffer, &[9.0; SCORE_STRIDE_SAMPLES], AUDIO_BUFFER_MAX);
        assert_eq!(drained, SCORE_STRIDE_SAMPLES);
        assert_eq!(buffer.len(), AUDIO_BUFFER_MAX);
        // The window below is the one a step reads. Its first sample is older
        // than this push by the stride, so it survived only because the cap keeps
        // a window *plus* a stride.
        let window = &buffer[buffer.len() - listen::WINDOW_SAMPLES..];
        let start = AUDIO_BUFFER_MAX - listen::WINDOW_SAMPLES;
        assert!(
            start >= drained,
            "the window's first sample was still in the ring"
        );
        assert_eq!(window[0], first[drained + start]);
        assert_eq!(window[window.len() - 1], 9.0);
    }

    /// An enrolment with no discriminative negatives: a window along the
    /// prototype scores 1.0, so the rolling sum crosses the static threshold on
    /// the second window of a segment.
    fn enrolment() -> crate::wake::Enrollment {
        use crate::wake::{Calibration, Enrollment};
        let prototype = vec![1.0f32, 0.0];
        let calibration = Calibration::default();
        let utterances = vec![prototype; crate::wake::MIN_ENROLLMENT_UTTERANCES];
        Enrollment::build("wake".into(), &utterances, calibration, &[])
            .expect("an enrolment of identical utterances")
    }

    fn stream() -> DetectionStream {
        DetectionStream::warm(enrolment())
    }

    /// A group's decisions are applied in the order its windows arrived: the
    /// fire happens at the window whose score crosses the threshold, not at the
    /// end of the group, and the windows after it are dropped with the cooldown
    /// measured from the fire's own position.
    #[test]
    fn a_group_fires_at_its_own_window_and_drops_the_rest() {
        let mut stream = stream();
        stream.samples_fed = 500;
        let embeddings = vec![vec![1.0f32, 0.0]; 4];
        let at = [100u64, 200, 300, 400];

        assert_eq!(
            stream.apply(&embeddings, &at),
            Some(200),
            "one fire, at its own window"
        );
        // The second window's rolling sum is the first two scores, the fire's
        // position is 200 and 300 samples of the slice arrived after it: the
        // cooldown is that much of its 3 s already spent.
        assert_eq!(stream.cooldown_left, COOLDOWN_SAMPLES - 300);
        // The post-fire reset ran, so nothing accumulated from the two windows
        // the group dropped.
        assert!(stream.score_window.is_empty());
        assert_eq!(stream.segment_score_steps, 0);
    }

    /// Windows below the threshold are all scored, in order, and the rolling
    /// window keeps the last [`crate::wake::ROLLING_WINDOW_N`] of them.
    #[test]
    fn a_group_below_the_threshold_scores_every_window() {
        let mut stream = stream();
        stream.samples_fed = 500;
        // A window between this enrolment's reset line and its fire factor: it
        // stays in the rolling window, and three of them still miss the bar.
        let thresholds = stream.enrollment.thresholds;
        let calibration = &stream.enrollment.calibration;
        let floor = calibration.soft_floor(&thresholds);
        let fire = calibration.fire_soft(&thresholds);
        let soft = 0.5 * (thresholds.reset_ratio * fire + fire);
        let cosine = floor + (1.0 - floor) * soft;
        let embedding = vec![cosine, (1.0 - cosine * cosine).sqrt()];
        let embeddings = vec![embedding; 4];

        assert_eq!(stream.apply(&embeddings, &[100, 200, 300, 400]), None);
        assert_eq!(stream.segment_score_steps, 4);
        assert_eq!(stream.score_window.len(), crate::wake::ROLLING_WINDOW_N);
        assert_eq!(stream.cooldown_left, 0);
    }

    /// An embedder that answers every window with the enrolment's prototype, so
    /// each window scores 1.0 and fires, and records what it was asked for.
    struct Prototype {
        calls: Vec<usize>,
    }

    impl Embedder for Prototype {
        fn embed_group(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            self.calls.push(windows.len());
            Ok(vec![vec![1.0f32, 0.0]; windows.len()])
        }
    }

    /// A segment close clears the rolling scores and the adaptive state. A window
    /// that was already queued must be applied before that happens: scored after
    /// it, the window would see a state it was not produced in, and its fire
    /// decision would differ from the ungrouped stream's.
    #[test]
    fn a_segment_close_never_overtakes_a_queued_window() {
        let mut stream = stream();
        // One window queued, and the silence timeout one hop from closing.
        stream.pending.push(vec![0.0f32; listen::WINDOW_SAMPLES]);
        stream.pending_at.push(0);
        stream.segment_silence_hops = SEGMENT_TIMEOUT_HOPS - 1;
        let mut embed = Prototype { calls: Vec::new() };

        stream
            .feed(
                &vec![0.0f32; FRAME_LENGTH],
                &mut embed,
                Deadline::new(60),
                "test",
            )
            .expect("a silent frame");

        assert_eq!(embed.calls, vec![1], "the queued window was embedded once");
        // The close ran last: if it had run first, the window's score would still
        // be in the rolling window afterwards.
        assert_eq!(stream.segment_silence_hops, 0, "the segment closed");
        assert!(
            stream.score_window.is_empty(),
            "the queued window was scored before the reset, not into it"
        );
    }

    /// The group is an execution detail, not part of the measurement: the same
    /// queued windows, resolved one at a time and resolved as one group, make the
    /// same decision at the same window and arm the same cooldown.
    #[test]
    fn a_grouped_and_an_ungrouped_stream_decide_alike() {
        let mut ungrouped = stream();
        let mut grouped = DetectionStream::with_group(enrolment(), listen::GROUP);
        // The same four windows, scored at the same stream positions: the second
        // one's rolling sum reaches the threshold.
        let embeddings = vec![vec![1.0f32, 0.0]; 4];
        let at = [100u64, 200, 300, 400];
        for stream in [&mut ungrouped, &mut grouped] {
            stream.samples_fed = 500;
        }

        assert_eq!(ungrouped.apply(&embeddings, &at), Some(200));
        assert_eq!(grouped.apply(&embeddings, &at), Some(200));
        assert_eq!(grouped.cooldown_left, ungrouped.cooldown_left);
        assert!(grouped.score_window.is_empty() && ungrouped.score_window.is_empty());
    }

    /// A voiced signal whose VAD is speech throughout: the streams below differ
    /// only in when they score a window, so the audio must not differ at all.
    fn speech(seconds: usize) -> Vec<f32> {
        (0..seconds * listen::SAMPLE_RATE as usize)
            .map(|n| {
                let t = n as f32 / listen::SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.6
            })
            .collect()
    }

    /// Feed `audio` in the live microphone's own chunks and report where the
    /// stream fired, and how much audio it walked as speech.
    fn decisions<E: Embedder>(
        stream: &mut DetectionStream,
        embed: &mut E,
        audio: &[f32],
    ) -> (Vec<u64>, u64) {
        let deadline = Deadline::new(600);
        for chunk in audio.chunks(FRAME_LENGTH) {
            stream
                .feed(chunk, embed, deadline, "test")
                .expect("a chunk");
        }
        stream.flush(embed, deadline, "test").expect("a flush");
        let fires = stream.take_fire_positions();
        let hops = stream.take_vad_hops();
        (fires, hops)
    }

    /// A caller's boundary is where its group's tail is scored, and scoring it there must
    /// not move a decision: the same audio fed with flushes at two interior boundaries
    /// fires where one continuous feed does, at the same stream positions, for the sweep's
    /// group, the product's pair and the ungrouped loop. That is what lets an offline pass
    /// flush at the end of every file and of every silence gap without changing what it
    /// reports — a boundary can move *when* a fire is applied, never where it is.
    ///
    /// The cuts sit on the feed's own frame grid, so the two runs differ in nothing but the
    /// flushes: a boundary off that grid re-chunks the audio, which moves a scoring step by
    /// up to one frame on its own, with or without a flush.
    #[test]
    fn a_flushed_boundary_decides_where_one_continuous_feed_does() {
        let audio = speech(6);
        let cuts = [32 * FRAME_LENGTH, 64 * FRAME_LENGTH];
        // `boundaries` are the sample counts at which the caller stops and flushes; the
        // last segment always ends the audio.
        let run = |boundaries: &[usize], group: usize| {
            let mut stream = DetectionStream::with_group(enrolment(), group);
            let mut embed = Prototype { calls: Vec::new() };
            let deadline = Deadline::new(600);
            let mut start = 0;
            for end in boundaries.iter().copied().chain([audio.len()]) {
                for chunk in audio[start..end].chunks(FRAME_LENGTH) {
                    stream
                        .feed(chunk, &mut embed, deadline, "test")
                        .expect("a chunk");
                }
                stream.flush(&mut embed, deadline, "test").expect("a flush");
                start = end;
            }
            (stream.take_fire_positions(), stream.take_vad_hops())
        };
        for group in [1, listen::PAIR, listen::GROUP] {
            let (whole, whole_hops) = run(&[], group);
            let (cut, cut_hops) = run(&cuts, group);
            assert!(
                !whole.is_empty(),
                "the fixture has to fire for the comparison to mean anything"
            );
            assert_eq!(
                cut, whole,
                "group {group}: a flush at a caller's boundary moved a fire"
            );
            // The VAD clock may count the hops of one group per fire more, as it does for
            // any grouped form: a fire is applied when its group resolves.
            let per_fire = (group.max(1) * listen::SCORE_STRIDE_SAMPLES / HOP_LENGTH) as u64;
            assert!(
                cut_hops.abs_diff(whole_hops) <= whole.len() as u64 * per_fire,
                "group {group}: the boundary moved the VAD clock by {} hops",
                cut_hops.abs_diff(whole_hops)
            );
        }
    }

    /// Two windows scored in one pass make the ungrouped stream's decisions at the
    /// ungrouped stream's own positions. The pair is the product's form; what it
    /// may move is when a fire is *reported* (its neighbour's feed) and the VAD
    /// clock of the hops that arrived before the fire was applied — at most one
    /// stride per fire — and neither is a decision.
    #[test]
    fn a_paired_stream_decides_where_an_ungrouped_one_does() {
        let audio = speech(6);
        let mut ungrouped = DetectionStream::warm(enrolment());
        let mut paired = DetectionStream::with_group(enrolment(), listen::PAIR);
        let mut embed = Prototype { calls: Vec::new() };

        let (single, single_hops) = decisions(&mut ungrouped, &mut embed, &audio);
        let (two, two_hops) = decisions(&mut paired, &mut embed, &audio);
        assert!(
            !single.is_empty(),
            "the fixture has to fire for the comparison to mean anything"
        );
        assert_eq!(two, single, "the same fires at the same stream positions");
        assert_eq!(paired.samples_fed(), ungrouped.samples_fed());
        // The paired stream walked the hops between a fire and the feed that
        // brought its neighbour as speech; the ungrouped one suppressed them.
        let stride_hops = (SCORE_STRIDE_SAMPLES / HOP_LENGTH) as u64;
        assert!(
            two_hops >= single_hops && two_hops - single_hops <= single.len() as u64 * stride_hops,
            "the VAD clock may grow by at most one stride per fire ({} against {single_hops})",
            two_hops
        );
    }

    /// The pair's decisions on real audio, not a fixture's: the pinned voice, streamed
    /// in the microphone's own chunks through an ungrouped stream and through the
    /// product's paired one, with the real encoder behind both. The synthetic test
    /// above shows the pairing cannot move a decision when the embeddings are equal;
    /// this one measures that on real speech the pair pass's own difference from a lone
    /// pass is not enough to move one — what it would find is a score close enough to a
    /// threshold for that difference to flip it.
    ///
    /// `#[ignore]`d: it compiles both front ends and needs the ASR pair and the voice.
    /// Run with `MAHBOT_MODELS_DIR=… MAHBOT_VOICE_WAV=… cargo test -p mahbot-inference
    /// --release --lib a_real_audio_paired_stream -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the ASR pair and the pinned voice; compiles both front ends"]
    fn a_real_audio_paired_stream_decides_where_an_ungrouped_one_does() {
        if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
            return;
        }
        let Ok(voice) = crate::loader::voice_wav() else {
            eprintln!("skipped: {} is not set", crate::loader::VOICE_WAV_ENV);
            return;
        };
        let pair = match crate::loader::Pair::of_role(crate::loader::Role::Recognition) {
            Ok(pair) => std::sync::Arc::new(pair),
            Err(error) => {
                eprintln!("skipped: {error:#}");
                return;
            }
        };
        let path = voice.display().to_string();
        let mut listener = listen::Listener::new(pair);

        // An enrolment of this voice: five windows one scoring stride apart, through
        // the ear's own one-window front end, so both streams score against it.
        let mut utterances = Vec::new();
        for i in 0..crate::wake::MIN_ENROLLMENT_UTTERANCES {
            let at = crate::stand::WAKE_OFFSET + i * listen::SCORE_STRIDE_SAMPLES;
            let window = crate::host::wav_mono_16k(&path, at, listen::WINDOW_SAMPLES)
                .expect("a window of the pinned voice");
            utterances.push(listener.embed(&window).expect("the window embeds"));
        }
        let enrollment = crate::wake::Enrollment::build(
            "wake".into(),
            &utterances,
            crate::wake::Calibration::default(),
            &[],
        )
        .expect("five utterances are an enrolment");

        // The whole file four times over, a second of digital silence between passes:
        // the 3 s audio-clock cooldown has to run out for the fixture to fire again.
        let pass = crate::host::wav_mono_16k(&path, 0, usize::MAX).expect("the pinned voice");
        let mut audio = Vec::new();
        for repeated in 0..4 {
            if repeated > 0 {
                audio.extend(vec![0.0f32; listen::SAMPLE_RATE as usize]);
            }
            audio.extend_from_slice(&pass);
        }

        // Both fronts on the one listener, which is what the product does.
        let mut ungrouped = DetectionStream::warm(enrollment.clone());
        let mut paired = DetectionStream::with_group(enrollment, listen::PAIR);
        let (single, single_hops) = decisions(&mut ungrouped, &mut listener, &audio);
        let (two, two_hops) = decisions(&mut paired, &mut listener, &audio);
        eprintln!(
            "ungrouped: {} fires at {:?} · {} speech hops; paired: {} fires at {:?} · {} speech hops",
            single.len(),
            single,
            single_hops,
            two.len(),
            two,
            two_hops
        );

        assert!(
            !single.is_empty(),
            "the pinned voice has to fire for the comparison to mean anything"
        );
        assert_eq!(
            two, single,
            "the two-window pass makes the ungrouped stream's fires at its own positions — a \
             difference is a decision the pair pass's embedding moved, not the pairing"
        );
        assert_eq!(paired.samples_fed(), ungrouped.samples_fed());
        let stride_hops = (SCORE_STRIDE_SAMPLES / HOP_LENGTH) as u64;
        assert!(
            two_hops >= single_hops && two_hops - single_hops <= single.len() as u64 * stride_hops,
            "the VAD clock may grow by at most one stride per fire ({} against {single_hops})",
            two_hops
        );
    }

    /// The pair waits for its neighbour and nothing more: a stream that has fed
    /// everything and not flushed holds at most one window, and the flush is what
    /// scores it.
    #[test]
    fn a_paired_stream_holds_one_window_for_its_neighbour() {
        let audio = speech(3);
        let mut stream = DetectionStream::with_group(enrolment(), listen::PAIR);
        let mut embed = Prototype { calls: Vec::new() };
        let deadline = Deadline::new(600);
        for chunk in audio.chunks(FRAME_LENGTH) {
            stream
                .feed(chunk, &mut embed, deadline, "test")
                .expect("a chunk");
            assert!(
                stream.pending.len() < listen::PAIR,
                "a full pair is scored where it fills, so one window is the most that waits"
            );
        }
        let waiting = stream.pending.len();
        stream.flush(&mut embed, deadline, "test").expect("a flush");
        assert!(stream.pending.is_empty(), "the flush scored what waited");
        assert!(
            waiting <= 1,
            "the pair holds one window, never a group's worth: {waiting}"
        );
    }
}
