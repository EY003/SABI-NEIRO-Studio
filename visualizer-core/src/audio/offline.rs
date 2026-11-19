//! Offline (file-based) analysis.
//!
//! Child module of `audio` on purpose: it can use the private `AudioProcessor`,
//! `DcBlocker` and `downmix` without changing their visibility. Add
//! `pub mod offline;` near the top of audio.rs.
//!
//! Determinism: `AudioProcessor::process` only depends on (history, settings, dt,
//! capture_instant). We feed it a fixed `dt` and a synthetic `Instant`
//! (base + step/rate), so the same file + settings always give the same frames.
//! Step `i` represents time `i / rate_hz`; its FFT window ENDS at that time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::{downmix, AudioProcessor, BarFrame, DcBlocker};
use crate::config::AudioSettings;

pub struct AnalysisTrack {
    pub frames: Vec<BarFrame>,
    pub rate_hz: f32,
    pub sample_rate: u32,
    pub duration_s: f32,
    /// Settings the track was built with; compare with `==` to know when to rebuild.
    pub settings: AudioSettings,
}

/// Analysis rate for a given output fps: fps times the smallest integer that reaches
/// >= 60 Hz (24 -> 72, 30 -> 60, 59.94 -> 59.94). Every output frame then lands exactly
/// on an analysis step (step = frame * multiple), so render needs no interpolation.
pub fn analysis_rate_for_fps(fps: f32) -> f32 {
    let fps = fps.max(1.0);
    let mult = (60.0 / fps - 0.01).ceil().max(1.0);
    fps * mult
}

impl AnalysisTrack {
    pub fn step_count(&self) -> usize { self.frames.len() }
    pub fn time_of(&self, step: usize) -> f32 { step as f32 / self.rate_hz }
    pub fn step_at(&self, t: f32) -> usize {
        ((t.max(0.0) * self.rate_hz).round() as usize).min(self.frames.len().saturating_sub(1))
    }

    /// Same prev/next blend the live loop does, for preview times between steps.
    pub fn frame_at(&self, t: f32) -> BarFrame {
        let last = self.frames.len().saturating_sub(1);
        let pos = (t.max(0.0) * self.rate_hz).min(last as f32);
        let i = pos.floor() as usize;
        if i >= last { return self.frames[last].clone(); }
        self.frames[i].lerp_toward(&self.frames[i + 1], pos - i as f32)
    }
}

/// Analyse interleaved samples. Returns `None` if cancelled.
pub fn analyze(
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
    settings: &AudioSettings,
    rate_hz: f32,
    cancel: &AtomicBool,
    mut progress: impl FnMut(f32),
) -> Option<AnalysisTrack> {
    let ch = channels.max(1);
    let n = samples.len() / ch;
    let sr = sample_rate as f32;

    // Same chain as the live path: channel downmix, then DC blocker.
    let mut mono: Vec<f32> = (0..n)
        .map(|i| downmix(&samples[i * ch..(i + 1) * ch], settings.channel_mix))
        .collect();
    if settings.dc_block {
        DcBlocker::new(sr).apply(&mut mono);
    }

    let rate = rate_hz.max(1.0);
    let dt = 1.0 / rate;
    let duration_s = n as f32 / sr;
    let steps = (duration_s * rate).ceil() as usize + 1;
    // Zero tail so the last windows decay to silence like the live idle path.
    mono.extend(std::iter::repeat(0.0).take((sr / rate).ceil() as usize + 2));

    let cap = (settings.fft_size * 4).max(8192);
    let mut proc = AudioProcessor::new(sr);
    let base = Instant::now();
    let mut frames: Vec<BarFrame> = Vec::with_capacity(steps);

    for i in 0..steps {
        if i % 64 == 0 {
            if cancel.load(Ordering::Relaxed) { return None; }
            progress(i as f32 / steps as f32);
        }
        let t = i as f64 / rate as f64;
        let end = ((t * sample_rate as f64).round() as usize).min(mono.len());
        let start = end.saturating_sub(cap);
        let at = base + Duration::from_secs_f64(t);
        let frame = proc
            .process(&mono[start..end], settings, dt, at)
            .or_else(|| frames.last().cloned())
            .unwrap_or_else(|| BarFrame::zeros(settings.num_bars));
        frames.push(frame);
    }
    progress(1.0);
    Some(AnalysisTrack { frames, rate_hz: rate, sample_rate, duration_s, settings: settings.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates() {
        assert_eq!(analysis_rate_for_fps(60.0), 60.0);
        assert_eq!(analysis_rate_for_fps(30.0), 60.0);
        assert_eq!(analysis_rate_for_fps(24.0), 72.0);
        assert!((analysis_rate_for_fps(59.94) - 59.94).abs() < 1e-3);
    }

    #[test]
    fn sine_is_deterministic_and_nonzero() {
        let sr = 48_000u32;
        let s: Vec<f32> = (0..sr as usize * 2)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let st = AudioSettings::default();
        let cancel = AtomicBool::new(false);
        let a = analyze(&s, 1, sr, &st, 60.0, &cancel, |_| {}).unwrap();
        let b = analyze(&s, 1, sr, &st, 60.0, &cancel, |_| {}).unwrap();
        assert_eq!(a.frames.len(), b.frames.len());
        let mid = a.frames.len() / 2;
        assert_eq!(a.frames[mid].values, b.frames[mid].values);
        assert!(a.frames[mid].values.iter().any(|&v| v > 0.05));
    }
}
