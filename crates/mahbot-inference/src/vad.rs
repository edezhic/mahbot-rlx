// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The voice-activity detector both the wake path and the transcription path
//! decide on: one `earshot` predictor at 256-sample hops, and the threshold that
//! turns its score into a decision.
//!
//! The detector carries its own state across hops, so onsets and offsets are
//! judged on one continuous stream — a caller that reset it per window would
//! answer differently at every cut.

/// Samples per VAD frame (the frame length a live microphone delivers).
pub const FRAME_LENGTH: usize = 512;

/// Samples per VAD hop — exactly one `earshot` frame.
pub const HOP_LENGTH: usize = 256;

/// VAD decision threshold: at or above is speech.
pub const VAD_THRESHOLD: f32 = 0.5;

/// One VAD decision. `samples` is chunked at 256 and a partial tail is
/// zero-padded; every sample is clamped to [-1, 1] first, because synthesized
/// audio can leave the range and `earshot` asserts on it.
pub fn is_speech(detector: &mut earshot::Detector, samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let mut speech = false;
    let mut frame = [0.0f32; HOP_LENGTH];
    for chunk in samples.chunks(HOP_LENGTH) {
        frame.fill(0.0);
        for (slot, &sample) in frame.iter_mut().zip(chunk) {
            *slot = sample.clamp(-1.0, 1.0);
        }
        if detector.predict_f32(&frame) >= VAD_THRESHOLD {
            speech = true;
        }
    }
    speech
}

/// The VAD-positive hops of `pcm`, concatenated — the "gated speech" the
/// calibration path encodes. The detector carries across hops, so onsets and
/// offsets are judged on the continuous stream.
pub fn vad_gate(pcm: &[f32], detector: &mut earshot::Detector) -> Vec<f32> {
    let mut speech = Vec::new();
    for hop in pcm.chunks(HOP_LENGTH) {
        if is_speech(detector, hop) {
            speech.extend_from_slice(hop);
        }
    }
    speech
}

/// The per-hop speech mask of a whole input, one entry per complete hop. The
/// samples past the last whole hop are not decided and belong to the caller's
/// last region. This is the detector's own answer, with no hangover and no
/// merging: [`crate::transcribe`] owns those.
pub fn speech_mask(pcm: &[f32], detector: &mut earshot::Detector) -> Vec<bool> {
    pcm.chunks_exact(HOP_LENGTH)
        .map(|hop| is_speech(detector, hop))
        .collect()
}
