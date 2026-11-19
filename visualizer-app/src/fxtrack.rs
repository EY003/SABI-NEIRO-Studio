//! Reactive-effects state for file mode.
//!
//! `FxState` is stateful (envelopes, chroma smoothing, the Dynamics histogram, the beat
//! clock), so it cannot be evaluated at an arbitrary time. We step it once per analysis
//! frame, keep a clone every CHECKPOINT_EVERY steps, and a seek restores the nearest
//! checkpoint and replays at most CHECKPOINT_EVERY steps (microseconds).
//! Export walks the steps in order and never needs checkpoints.

use visualizer_core::audio::offline::AnalysisTrack;
use visualizer_core::config::{AppConfig, AudioSettings, MotionSettings, ReactiveSettings};

use crate::reactive::FxState;

pub const CHECKPOINT_EVERY: usize = 120;

#[derive(Clone, PartialEq)]
struct FxKey {
    reactive: ReactiveSettings,
    audio: AudioSettings,
    motion: MotionSettings,
    speed: f32,
}
impl FxKey {
    fn of(cfg: &AppConfig) -> Self {
        Self { reactive: cfg.reactive.clone(), audio: cfg.audio.clone(),
               motion: cfg.color_change_motion.clone(), speed: cfg.color_change_speed }
    }
}

/// One simulation step: identical call order to the live `update()`.
pub fn step(fx: &mut FxState, tr: &AnalysisTrack, cfg: &AppConfig, i: usize) {
    let f = &tr.frames[i];
    let t = i as f32 / tr.rate_hz;
    let dt = 1.0 / tr.rate_hz;
    fx.on_audio_frame(f, &cfg.reactive);
    fx.advance_clock(dt, t, f.bpm, cfg);
    fx.update(Some(f), f.phase, f.beat_in, dt, t, cfg, f.values.len());
}

pub struct FxTrack {
    /// checkpoints[k] = state BEFORE step k * CHECKPOINT_EVERY.
    checkpoints: Vec<FxState>,
    key: FxKey,
}

impl FxTrack {
    /// Returns `None` if `cancel` returns true. Call again whenever `is_valid_for` is false.
    pub fn build(tr: &AnalysisTrack, cfg: &AppConfig, mut cancel: impl FnMut(f32) -> bool) -> Option<Self> {
        let mut fx = FxState::new();
        let mut checkpoints = Vec::with_capacity(tr.step_count() / CHECKPOINT_EVERY + 1);
        for i in 0..tr.step_count() {
            if i % CHECKPOINT_EVERY == 0 {
                if cancel(i as f32 / tr.step_count() as f32) { return None; }
                checkpoints.push(fx.clone());
            }
            step(&mut fx, tr, cfg, i);
        }
        Some(Self { checkpoints, key: FxKey::of(cfg) })
    }

    pub fn is_valid_for(&self, cfg: &AppConfig) -> bool { self.key == FxKey::of(cfg) }

    /// State after stepping through `target` (the state used to draw that frame).
    pub fn state_at(&self, tr: &AnalysisTrack, cfg: &AppConfig, target: usize) -> FxState {
        let k = (target / CHECKPOINT_EVERY).min(self.checkpoints.len().saturating_sub(1));
        let mut fx = self.checkpoints[k].clone();
        for i in k * CHECKPOINT_EVERY..=target.min(tr.step_count() - 1) {
            step(&mut fx, tr, cfg, i);
        }
        fx
    }
}

/// Keeps a running state during playback so sequential frames cost one step each;
/// a jump (seek) falls back to the checkpoint path.
pub struct FxCursor {
    pub fx: FxState,
    /// Next step to apply (`fx` currently reflects step `next - 1`).
    next: usize,
}
impl FxCursor {
    pub fn new() -> Self { Self { fx: FxState::new(), next: 0 } }

    pub fn advance_to(&mut self, track: &FxTrack, tr: &AnalysisTrack, cfg: &AppConfig, target: usize) {
        let target = target.min(tr.step_count().saturating_sub(1));
        if self.next > 0 && target + 1 == self.next { return; } // same frame
        if target + 1 > self.next && target + 1 - self.next <= CHECKPOINT_EVERY {
            for i in self.next..=target { step(&mut self.fx, tr, cfg, i); }
        } else {
            self.fx = track.state_at(tr, cfg, target);
        }
        self.next = target + 1;
    }
}
