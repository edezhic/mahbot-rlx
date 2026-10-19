// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side pieces the live paths still need: RoPE tables at global positions,
//! the sampler, and a RIFF/WAVE reader for the recorded 16 kHz asset.

use anyhow::Result;

/// The same, for rows that keep **global** positions `from..from + rows` —
/// what a window of a cached sequence needs: RoPE is applied at the position a
/// row has in the whole sequence, not at its offset inside the window. This is
/// f64 on purpose: the body's cached walk can reach thousands of positions, and
/// the graph's f32 synthetic tables (for a whole wave window) would drift.
pub fn rope_tables_at(from: usize, rows: usize, hd: usize, base: f64) -> (Vec<f64>, Vec<f64>) {
    let half = hd / 2;
    let mut cos = vec![0f64; rows * half];
    let mut sin = vec![0f64; rows * half];
    for r in 0..rows {
        let p = from + r;
        for i in 0..half {
            let theta = p as f64 / base.powf(2.0 * i as f64 / hd as f64);
            cos[r * half + i] = theta.cos();
            sin[r * half + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn threefry2x32_20(c: (u32, u32), k: (u32, u32)) -> (u32, u32) {
    const KS_PARITY: u32 = 0x1BD1_1BDA;
    let ks = [k.0, k.1, k.0 ^ k.1 ^ KS_PARITY];
    let (mut x0, mut x1) = (c.0.wrapping_add(ks[0]), c.1.wrapping_add(ks[1]));
    let mix = |x0: &mut u32, x1: &mut u32, r: u32| {
        *x0 = x0.wrapping_add(*x1);
        *x1 = x1.rotate_left(r) ^ *x0;
    };
    for (i, group) in [
        [13u32, 15, 26, 6],
        [17, 29, 16, 24],
        [13, 15, 26, 6],
        [17, 29, 16, 24],
        [13, 15, 26, 6],
    ]
    .iter()
    .enumerate()
    {
        for r in group {
            mix(&mut x0, &mut x1, *r);
        }
        // Injection `i + 1`: the key words walk the schedule, and the round's own
        // number is added to the second word — exactly the shader's injections.
        x0 = x0.wrapping_add(ks[(i + 1) % 3]);
        x1 = x1.wrapping_add(ks[(i + 2) % 3]).wrapping_add(i as u32 + 1);
    }
    (x0, x1)
}

/// The engine's draw, on the host: temperature → softmax → top-k/top-p selection
/// → Gumbel-max over Threefry-2×32-20 — the arithmetic `Op::Sample` runs in its
/// shader, so the same logits and seed give the same id without a device trip.
pub fn draw(logits: &[f32], top_k: usize, top_p: f32, temperature: f32, seed: u64) -> usize {
    let inv_temp = 1.0 / temperature.max(1e-6);
    let mut p: Vec<f32> = logits.iter().map(|v| *v * inv_temp).collect();
    let m = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in p.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in p.iter_mut() {
        *v *= inv;
    }
    if top_k > 0 || (top_p < 1.0 && top_p > 0.0) {
        let k_limit = if top_k > 0 {
            top_k.min(p.len())
        } else {
            p.len()
        };
        // The repeated maximum the sampler used to scan for is a stable
        // descending order (`p` desc, then index asc): selecting the first
        // `k_limit` with a partial sort and sorting only those gives the same
        // prefix, and the top-p stop below cuts that prefix at the same place.
        let order = |a: &usize, b: &usize| {
            p[*b]
                .partial_cmp(&p[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        };
        let mut ranked: Vec<usize> = (0..p.len()).collect();
        if k_limit < ranked.len() {
            ranked.select_nth_unstable_by(k_limit, order);
            ranked.truncate(k_limit);
        }
        ranked.sort_unstable_by(order);
        let mut cum = 0f32;
        for &i in &ranked {
            let v = p[i];
            p[i] = -v - 1e-30;
            cum += v;
            if top_p < 1.0 && cum >= top_p {
                break;
            }
        }
        let mut new_sum = 0f32;
        for v in p.iter_mut() {
            if *v < 0.0 {
                *v = -*v - 1e-30;
                new_sum += *v;
            } else {
                *v = 0.0;
            }
        }
        let inv = 1.0 / new_sum.max(1e-12);
        for v in p.iter_mut() {
            *v *= inv;
        }
    }
    let raw = (seed as u32, (seed >> 32) as u32);
    let key = threefry2x32_20((0, 0), raw);
    let (mut best, mut picked) = (f32::NEG_INFINITY, 0usize);
    for (i, pr) in p.iter().enumerate() {
        if *pr <= 0.0 {
            continue;
        }
        let (r0, _) = threefry2x32_20((0, i as u32), key);
        let u = (r0 >> 8) as f32 / 16_777_216.0;
        let g = -(-u.max(1e-30).ln()).ln();
        let score = pr.ln() + g;
        if score > best {
            best = score;
            picked = i;
        }
    }
    picked
}

/// Write mono PCM16 WAV: `samples` at `rate` Hz.
pub fn write_wav16(path: &str, samples: &[f32], rate: u32) -> Result<()> {
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &v in samples {
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// `offset` is in output samples; `max_samples` limits the output.
pub fn wav_mono(path: &str, rate: u32, offset: usize, max_samples: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(anyhow::anyhow!("`{path}` is not a RIFF/WAVE file"));
    }
    let mut at = 12usize;
    let (mut format, mut channels, mut src_rate, mut bits) = (1u16, 1u16, 16000u32, 16u16);
    let mut data: Option<&[u8]> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = bytes.get(at + 8..at + 8 + len).unwrap_or(&[]);
        match id {
            b"fmt " if body.len() >= 16 => {
                format = u16::from_le_bytes(body[0..2].try_into().unwrap());
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                src_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            }
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let data = data.ok_or_else(|| anyhow::anyhow!("`{path}` has no data chunk"))?;
    if channels == 0 || !((format == 1 && bits == 16) || (format == 3 && bits == 32)) {
        return Err(anyhow::anyhow!(
            "`{path}` is not PCM16 or float32 (format={format}, bits={bits}, channels={channels})"
        ));
    }
    if src_rate == 0 || rate == 0 {
        return Err(anyhow::anyhow!("`{path}` declares a zero sample rate"));
    }
    let bytes_per = if format == 3 { 4 } else { 2 };
    let frames = data.len() / (bytes_per * channels as usize);
    let sample = |i: usize| -> f32 {
        let j = i * channels as usize;
        if format == 3 {
            f32::from_le_bytes([
                data[4 * j],
                data[4 * j + 1],
                data[4 * j + 2],
                data[4 * j + 3],
            ])
        } else {
            i16::from_le_bytes([data[2 * j], data[2 * j + 1]]) as f32 / 32768.0
        }
    };
    let step = src_rate as f64 / rate as f64;
    let first = offset as f64 * step;
    let mut out = Vec::with_capacity(max_samples.min(frames));
    let mut o = 0usize;
    while out.len() < max_samples {
        let src = first + o as f64 * step;
        let i = src.floor() as usize;
        if i >= frames {
            break;
        }
        let frac = (src - i as f64) as f32;
        let s = if i + 1 < frames {
            sample(i) * (1.0 - frac) + sample(i + 1) * frac
        } else {
            sample(i)
        };
        out.push(s);
        o += 1;
    }
    Ok(out)
}

pub fn wav_mono_16k(path: &str, offset: usize, max_samples: usize) -> Result<Vec<f32>> {
    wav_mono(path, 16_000, offset, max_samples)
}

/// Linear-interpolation resample of mono PCM (the same arithmetic as
/// [`wav_mono`]'s reader with no offset and no sample limit). Returns `samples`
/// unchanged when the rates are equal.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || from == 0 || to == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let step = from as f64 / to as f64;
    let out_len = (samples.len() as f64 / step).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for o in 0..out_len {
        let src = o as f64 * step;
        let i = src.floor() as usize;
        if i >= samples.len() {
            break;
        }
        let frac = (src - i as f64) as f32;
        let s = if i + 1 < samples.len() {
            samples[i] * (1.0 - frac) + samples[i + 1] * frac
        } else {
            samples[i]
        };
        out.push(s);
    }
    out
}

fn mel_filters_slaney(bins: usize, n_freq: usize, sample_rate: f64, f_max: f64) -> Vec<f32> {
    // librosa.filters.mel(htk=False, norm="slaney"), the reference's filter bank.
    let mel = |f: f64| {
        if f < 1000.0 {
            3.0 * f / 200.0
        } else {
            15.0 + 27.0 * (f / 1000.0).ln() / 6.4f64.ln()
        }
    };
    let hz = |m: f64| {
        if m < 15.0 {
            200.0 * m / 3.0
        } else {
            1000.0 * (6.4f64.ln() * (m - 15.0) / 27.0).exp()
        }
    };
    let (m_lo, m_hi) = (mel(0.0), mel(f_max));
    let points: Vec<f64> = (0..bins + 2)
        .map(|i| hz(m_lo + (m_hi - m_lo) * i as f64 / (bins + 1) as f64))
        .collect();
    let mut fb = vec![0f32; n_freq * bins];
    for m in 0..bins {
        let (lo, mid, hi) = (points[m], points[m + 1], points[m + 2]);
        let area = if hi > lo { 2.0 / (hi - lo) } else { 0.0 };
        for f in 0..n_freq {
            let freq = f as f64 * sample_rate / ((n_freq - 1) * 2) as f64;
            let v = if freq >= lo && freq <= mid {
                (freq - lo) / (mid - lo)
            } else if freq > mid && freq <= hi {
                (hi - freq) / (hi - mid)
            } else {
                0.0
            };
            fb[f * bins + m] = (v * area) as f32;
        }
    }
    fb
}

/// 24 kHz log-mel for the speaker encoder: the reference's `mel_spectrogram`
/// (n_fft 1024, hop 256, 128 bins, 0–12 kHz, reflect padding, Slaney filters,
/// amplitude `sqrt(re²+im²+1e-9)` before the filter bank, natural log). This is a
/// separate front from [`crate::graphs::MelStage`]: that one is HTK/power for the
/// recognition encoder.
pub fn mel_24k(samples: &[f32]) -> (Vec<f32>, usize) {
    const N: usize = 1024;
    const HOP: usize = 256;
    const BINS: usize = 128;
    let pad = 384usize;
    let mut padded = vec![0f32; samples.len() + 2 * pad];
    for i in 0..padded.len() {
        let src = i as isize - pad as isize;
        let j = if src < 0 {
            (-src) as usize
        } else if src as usize >= samples.len() {
            2 * samples.len() - 2 - src as usize
        } else {
            src as usize
        };
        padded[i] = samples[j.min(samples.len() - 1)];
    }
    let frames = (padded.len().saturating_sub(N)) / HOP + 1;
    let window: Vec<f32> = (0..N)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos()) as f32)
        .collect();
    let n_freq = N / 2 + 1;
    let fb = mel_filters_slaney(BINS, n_freq, 24_000.0, 12_000.0);
    let mut out = vec![0f32; BINS * frames];
    for t in 0..frames {
        let mut re = vec![0f32; n_freq];
        let mut im = vec![0f32; n_freq];
        for k in 0..n_freq {
            let mut sr = 0f32;
            let mut si = 0f32;
            for i in 0..N {
                let a = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / N as f64;
                let x = padded[t * HOP + i] * window[i];
                sr += x * a.cos() as f32;
                si -= x * a.sin() as f32;
            }
            re[k] = sr;
            im[k] = si;
        }
        for m in 0..BINS {
            let mut sum = 0f32;
            for k in 0..n_freq {
                sum += (re[k] * re[k] + im[k] * im[k] + 1e-9).sqrt() * fb[k * BINS + m];
            }
            out[m * frames + t] = sum.max(1e-5).ln();
        }
    }
    (out, frames)
}

#[cfg(test)]
mod tests {
    use super::resample;

    #[test]
    fn the_same_rate_returns_the_input_unchanged() {
        let samples = [0.25f32, -0.5, 1.0, 0.0];
        assert_eq!(resample(&samples, 16_000, 16_000), samples);
    }

    #[test]
    fn twenty_four_k_to_sixteen_k_keeps_three_samples_in_two() {
        let samples: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = resample(&samples, 24_000, 16_000);
        assert_eq!(out.len(), 16);
        // 24·2/3 = 16 output samples, at source positions `o · 1.5`.
        assert_eq!(out[1], 1.5);
        assert_eq!(out[2], 3.0);
    }
}
