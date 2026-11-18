// ═══════════════════════════════════════════════════════════════
// reactive.rs — audio-reactive visual effects
//
// Lives next to main.rs (`mod reactive;`).  It turns the analysis fields
// that used to feed only the diagnostics window (beat_envelope, phase,
// onsets, chroma, pitch, band_levels, waveform, spectrum_row, rms, peak …)
// into per-bar modifiers that main.rs applies while drawing:
//
//   * geometry  : height multiplier, horizontal shift, width multiplier,
//                 peak-cap size
//   * colour    : overlay / dim / additive layers, global hue rotation,
//                 dynamics saturation + brightness
//   * overlays  : raw-spectrum ghost line, dominant-pitch cap / outline
//   * clock     : beat-locked time for ColorMode::Animated
//
// Every effect is individually switchable (ReactiveSettings in config.rs)
// and nothing here touches the audio analysis itself.
// ═══════════════════════════════════════════════════════════════

use std::f32::consts::TAU;
use std::time::Instant;

use eframe::egui;
use visualizer_core::audio::{compute_bar_centers_hz, BarFrame};
use visualizer_core::config::{
    AppConfig, BandColorMode, BarInterpolation, FlashMode, FrequencyDirection, FrequencyScaling,
    GhostSource, ReactiveSettings, SweepMode, VisualizerDirection,
};

use super::{
    bar_vertex, checkbox_reset, color_edit_reset, combo_reset, draw_peak_hold,
    frequency_index, hash_u32, help_text, lerp_color, section_header, slider_f32, slider_u8,
    smoothstep, SettingsState, VisualizerApp,
};

/// Effects may push a bar this many pixels past the configured maximum.
/// The overlay window has 32 px of padding around the bars, so this is safe.
pub const FX_HEADROOM_PX: f32 = 24.0;

/// Geometric centres of the four `band_levels` (20–100, 100–500, 500–2k, 2k–8k Hz).
const BAND_CENTERS_HZ: [f32; 4] = [44.7, 223.6, 1000.0, 4000.0];

// ───────────────────────────────────────────────────────────────
// Small helpers
// ───────────────────────────────────────────────────────────────

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

#[inline]
fn rgb01(c: [u8; 3]) -> [f32; 3] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

/// 0..1 weight of how close `f` is to the band [lo, hi] (in octaves).
fn region_weight(f: f32, lo: f32, hi: f32, soft_oct: f32) -> f32 {
    let d = if f < lo { (lo / f.max(1.0)).log2() }
            else if f > hi { (f / hi).log2() }
            else { 0.0 };
    1.0 - smoothstep(0.0, soft_oct.max(0.01), d)
}

/// "Source-over" compositing of a colour layer into an RGBA accumulator.
#[inline]
fn over(dst: &mut [f32; 4], rgb: [f32; 3], a: f32) {
    let a = a.clamp(0.0, 1.0);
    if a <= 0.0 { return; }
    let out_a = a + dst[3] * (1.0 - a);
    if out_a <= 1e-6 { return; }
    for k in 0..3 {
        dst[k] = (rgb[k] * a + dst[k] * dst[3] * (1.0 - a)) / out_a;
    }
    dst[3] = out_a;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32) % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Tracks a slowly-adapting [lo, hi] range of a signal and returns the
/// current value normalised into it. The memory is `tau` seconds, so the
/// range follows the music instead of being pinned by a single outlier.
#[derive(Clone, Copy, Default)]
struct AdaptiveRange { lo: f32, hi: f32, init: bool }
impl AdaptiveRange {
    fn update(&mut self, x: f32, dt: f32, tau: f32, min_span: f32) -> f32 {
        if !self.init {
            self.lo = x - min_span * 0.5;
            self.hi = x + min_span * 0.5;
            self.init = true;
        }
        let k = (dt / tau.max(0.1)).clamp(0.0, 1.0);
        if x < self.lo { self.lo = x } else { self.lo += (x - self.lo) * k }
        if x > self.hi { self.hi = x } else { self.hi += (x - self.hi) * k }
        let span = (self.hi - self.lo).max(min_span);
        ((x - self.lo) / span).clamp(0.0, 1.0)
    }
}

// ───────────────────────────────────────────────────────────────
// Dynamics ("velocity") analyser
//
// Produces `intensity` in 0..1: how hard the music is being played
// *relative to the track itself*. It blends three cues:
//   loudness   – percentile rank of the short-term level inside a decaying
//                histogram of the whole track so far (robust to outliers)
//   attack     – how sharply the level just rose (struck keys / plucks)
//   brightness – spectral centroid vs. its own recent range (harder
//                playing is brighter)
// Because everything is relative to the track's own history, a quiet solo
// piano piece uses the full range just like a loud rock track.
// ───────────────────────────────────────────────────────────────

const HIST_BINS: usize = 80; // 1 dB bins covering -80 .. 0 dBFS

pub struct Dynamics {
    hist: Vec<f32>,
    total: f32,
    loud_fast: f32,
    loud_slow: f32,
    attack_hold: f32,
    bright: AdaptiveRange,
    init: bool,
    pub intensity: f32,
}

impl Default for Dynamics {
    fn default() -> Self {
        Self {
            hist: vec![0.0; HIST_BINS],
            total: 0.0,
            loud_fast: -80.0,
            loud_slow: -80.0,
            attack_hold: 0.0,
            bright: AdaptiveRange::default(),
            init: false,
            intensity: 0.5,
        }
    }
}

impl Dynamics {
    /// Forget everything learned about the current track (call on track change).
    pub fn reset(&mut self) { *self = Dynamics::default(); }

    fn update(&mut self, f: &BarFrame, dt: f32, rs: &ReactiveSettings) {
        let db = 20.0 * f.rms.max(1e-6).log10();
        let active = db > -70.0;
        if !self.init {
            self.loud_fast = db;
            self.loud_slow = db;
            self.init = true;
        }
        self.loud_fast += (db - self.loud_fast) * (1.0 - (-dt / 0.08).exp());
        self.loud_slow += (db - self.loud_slow) * (1.0 - (-dt / 0.40).exp());

        // Attack: sudden rise of the fast level over the slow one, held briefly.
        let attack_n = ((self.loud_fast - self.loud_slow).max(0.0) / 9.0).clamp(0.0, 1.0);
        self.attack_hold = attack_n.max(self.attack_hold * (-dt / 0.7).exp());

        // Brightness: centroid of the (log-frequency) spectrum row.
        let (mut num, mut den) = (0.0f32, 0.0f32);
        let len = f.spectrum_row.len();
        for (i, &v) in f.spectrum_row.iter().enumerate() {
            let w = v.clamp(0.0, 1.0).powi(4);
            num += w * i as f32;
            den += w;
        }
        let centroid = if den > 1e-6 && len > 1 { num / den / (len - 1) as f32 } else { 0.5 };
        let tau = rs.dyn_memory_s.max(5.0);
        let bright_n = self.bright.update(centroid, dt, tau, 0.08);

        // Loudness percentile inside the decaying whole-track histogram.
        let decay = (-dt / tau).exp();
        for h in self.hist.iter_mut() { *h *= decay; }
        self.total *= decay;
        let bin = (self.loud_fast + 80.0).clamp(0.0, HIST_BINS as f32 - 0.001) as usize;
        if active {
            self.hist[bin] += dt;
            self.total += dt;
        }
        let rank = if self.total > 1e-3 {
            let below: f32 = self.hist[..bin].iter().sum::<f32>() + self.hist[bin] * 0.5;
            let conf = (self.total / 4.0).clamp(0.0, 1.0); // neutral during warm-up
            0.5 + (below / self.total - 0.5) * conf
        } else {
            0.5
        };

        let wl = rs.dyn_w_loudness.max(0.0);
        let wa = rs.dyn_w_attack.max(0.0);
        let wb = rs.dyn_w_brightness.max(0.0);
        let wsum = wl + wa + wb;
        let raw = if !active {
            0.0
        } else if wsum < 1e-3 {
            rank
        } else {
            (wl * rank + wa * self.attack_hold + wb * bright_n) / wsum
        };

        let c = rs.dyn_contrast.clamp(0.3, 4.0);
        let target = (0.5 + (raw - 0.5) * c).clamp(0.0, 1.0);
        let tau_s = if target > self.intensity { rs.dyn_attack_ms } else { rs.dyn_release_ms };
        let k = 1.0 - (-dt / (tau_s / 1000.0).max(0.005)).exp();
        self.intensity += (target - self.intensity) * k;
    }
}

// ───────────────────────────────────────────────────────────────
// FxState
// ───────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct FxState {
    // Layout cache (rebuilt when the bar layout or region softness changes)
    layout_key: Option<(usize, FrequencyScaling, u32, u32, u32)>,
    bar_hz: Vec<f32>,
    w_drum: [Vec<f32>; 3],
    bar_pc: Vec<i8>,

    // Event envelopes (set per audio frame, decayed per UI frame)
    spike_env: [f32; 3],
    flash_env: [f32; 3],
    snap_env: f32,
    flux_env: f32,
    prev_peak: f32,
    prev_capture: Option<Instant>,

    // Slow state
    band_range: [AdaptiveRange; 4],
    band_n: [f32; 4],
    chroma_s: [f32; 12],
    chroma_span: f32,
    chroma_min: f32,
    key_pc: usize,
    rot_deg: f32,
    pitch_ln: f32,
    pitch_w: f32,
    since_pitch: f32,
    wave_bar: Vec<f32>,

    pub dynamics: Dynamics,
    /// Clock handed to the colour code. Equals wall time unless beat-clock is on.
    pub sync_time: f32,

    // Per-frame outputs (indexed by *frequency* bar index)
    pub active: bool,
    pub color_active: bool,
    height_mul: Vec<f32>,
    peak_mul: Vec<f32>,
    x_shift: Vec<f32>,
    width_mul: Vec<f32>,
    overlay: Vec<[f32; 4]>,
    dim: Vec<f32>,
    add: Vec<[f32; 3]>,
    pub ghost: Vec<f32>,
    pub pitch_bar: Vec<f32>,
    pub pitch_w_public: f32,
    pub outline_scale: f32,
    // global colour transform
    dyn_on: bool,
    dyn_sat: f32,
    dyn_val: f32,
    hue_deg: f32,
}

impl FxState {
    pub fn new() -> Self {
        let mut s = FxState::default();
        s.outline_scale = 1.0;
        s
    }

    // ── per-audio-frame events ─────────────────────────────────
    /// Call once for every frame received from the audio thread, *before*
    /// interpolation, so one-frame onset flags are never missed.
    pub fn on_audio_frame(&mut self, f: &BarFrame, rs: &ReactiveSettings) {
        let dt = match self.prev_capture {
            Some(p) => f.capture_instant.saturating_duration_since(p).as_secs_f32().clamp(0.002, 0.25),
            None => 1.0 / 60.0,
        };
        self.prev_capture = Some(f.capture_instant);

        if f.onsets.kick  { self.spike_env[0] = 1.0; self.flash_env[0] = 1.0; }
        if f.onsets.snare { self.spike_env[1] = 1.0; self.flash_env[1] = 1.0; }
        if f.onsets.hat   { self.spike_env[2] = 1.0; self.flash_env[2] = 1.0; }

        let thr = rs.snap_threshold.clamp(0.05, 1.0);
        if f.peak >= thr && self.prev_peak < thr { self.snap_env = 1.0; }
        self.prev_peak = f.peak;

        if rs.enabled && rs.dynamics_color {
            self.dynamics.update(f, dt, rs);
        }
    }

    // ── colour clock (idea 6) ──────────────────────────────────
    /// Advances `sync_time`. With beat-clock on it runs at the song's tempo
    /// (so every MotionShape preset locks to the beat and follows
    /// Half/Double-Time); otherwise it simply mirrors wall-clock time, so
    /// switching the option on or off never makes the colours jump.
    pub fn advance_clock(&mut self, dt: f32, wall: f32, bpm: f32, cfg: &AppConfig) {
        let rs = &cfg.reactive;
        if rs.enabled && rs.beat_clock_colors {
            if bpm >= 20.0 {
                let m = &cfg.color_change_motion;
                let eff = ((0.05 + cfg.color_change_speed * 1.95) * m.speed).max(1e-3);
                let cycles_per_s = (bpm / 60.0) / rs.beat_clock_beats_per_cycle.max(0.25);
                self.sync_time += dt * cycles_per_s / eff;
            } else {
                self.sync_time += dt; // no tempo yet: wall-clock speed
            }
        } else {
            self.sync_time = wall;
        }
    }

    // ── layout ─────────────────────────────────────────────────
    fn ensure_layout(&mut self, a: &visualizer_core::config::AudioSettings, rs: &ReactiveSettings, n: usize) {
        let key = (n, a.frequency_scaling, a.min_frequency_hz.to_bits(),
                   a.max_frequency_hz.to_bits(), rs.spike_softness_oct.to_bits());
        if self.layout_key == Some(key) && self.bar_hz.len() == n { return; }
        self.layout_key = Some(key);

        let mut hz = compute_bar_centers_hz(n, a.frequency_scaling, a.min_frequency_hz, a.max_frequency_hz);
        if hz.len() != n {
            hz = (0..n)
                .map(|i| 30.0 * (16_000.0f32 / 30.0).powf((i as f32 + 0.5) / n.max(1) as f32))
                .collect();
        }
        const BANDS: [(f32, f32); 3] = [(20.0, 150.0), (150.0, 2000.0), (6000.0, 12000.0)];
        for d in 0..3 {
            self.w_drum[d] = hz.iter()
                .map(|&f| region_weight(f, BANDS[d].0, BANDS[d].1, rs.spike_softness_oct))
                .collect();
        }
        // Chroma only covers 200–4000 Hz (see audio.rs); other bars get no pitch class.
        self.bar_pc = hz.iter().map(|&f| {
            if (200.0..=4000.0).contains(&f) {
                let midi = 69.0 + 12.0 * (f / 440.0).log2();
                ((midi.round() as i32).rem_euclid(12)) as i8
            } else { -1 }
        }).collect();
        self.bar_hz = hz;
    }

    fn hz_to_pos(&self, hz: f32) -> Option<f32> {
        let n = self.bar_hz.len();
        if n == 0 || hz <= 0.0 { return None; }
        if n == 1 { return Some(0.0); }
        if hz < self.bar_hz[0] / 1.5 || hz > self.bar_hz[n - 1] * 1.5 { return None; }
        let i = self.bar_hz.partition_point(|&h| h <= hz);
        if i == 0 { return Some(0.0); }
        if i >= n { return Some((n - 1) as f32); }
        let (a, b) = (self.bar_hz[i - 1], self.bar_hz[i]);
        let t = ((hz / a).ln() / (b / a).ln().max(1e-6)).clamp(0.0, 1.0);
        Some((i - 1) as f32 + t)
    }

    /// Smoothly interpolated (log-frequency) normalised band level at `f` Hz.
    fn band_interp(&self, f: f32) -> f32 {
        let c = &BAND_CENTERS_HZ;
        if f <= c[0] { return self.band_n[0]; }
        if f >= c[3] { return self.band_n[3]; }
        for b in 0..3 {
            if f < c[b + 1] {
                let t = (f / c[b]).ln() / (c[b + 1] / c[b]).ln();
                return lerp(self.band_n[b], self.band_n[b + 1], t);
            }
        }
        self.band_n[3]
    }

    /// Triangular affinity (0..1) of frequency `f` to band `b`'s centre.
    fn band_affinity(b: usize, f: f32) -> f32 {
        let d = (f.max(1.0) / BAND_CENTERS_HZ[b]).ln().abs();
        (1.0 - d / 1.8).clamp(0.0, 1.0)
    }

    // ── per-UI-frame update ────────────────────────────────────
    /// `phase` and `beat_in` should already be extrapolated to "now".
    pub fn update(
        &mut self,
        f: Option<&BarFrame>,
        phase: f32,
        beat_in: f32,
        dt: f32,
        wall_time: f32,
        cfg: &AppConfig,
        n: usize,
    ) {
        let rs = &cfg.reactive;
        self.active = rs.any_active() && n > 0;
        self.outline_scale = 1.0;
        self.pitch_w_public = 0.0;
        if !self.active {
            self.color_active = false;
            self.dyn_on = false;
            self.ghost.clear();
            self.pitch_bar.clear();
            return;
        }

        self.ensure_layout(&cfg.audio, rs, n);

        macro_rules! reset { ($v:expr, $val:expr) => { $v.clear(); $v.resize(n, $val); }; }
        reset!(self.height_mul, 1.0f32);
        reset!(self.peak_mul, 1.0f32);
        reset!(self.x_shift, 0.0f32);
        reset!(self.width_mul, 1.0f32);
        reset!(self.overlay, [0.0f32; 4]);
        reset!(self.dim, 1.0f32);
        reset!(self.add, [0.0f32; 3]);
        reset!(self.pitch_bar, 0.0f32);
        self.ghost.clear();

        // Envelope decay
        let kdec = |ms: f32| (-dt / (ms / 1000.0).max(0.005)).exp();
        let ks = kdec(rs.spike_decay_ms);
        let kf = kdec(rs.flash_decay_ms);
        for e in self.spike_env.iter_mut() { *e *= ks; }
        for e in self.flash_env.iter_mut() { *e *= kf; }
        self.snap_env *= (-dt / 0.08).exp();

        let (bpm, rms, flux, bands, chroma, pitch_hz, beat_env) = match f {
            Some(f) => (f.bpm, f.rms, f.onsets.flux, f.band_levels, f.chroma,
                        f.dominant_pitch_hz, f.beat_envelope),
            None => (0.0, 0.0, 0.0, [0.0; 4], [0.0; 12], 0.0, 0.0),
        };
        let activity = smoothstep(0.001, 0.01, rms);
        let beat_ok = bpm >= 20.0;
        let phase = phase.rem_euclid(1.0);

        self.flux_env += (flux - self.flux_env) * (1.0 - (-dt / 0.05).exp());
        if rs.transient_snap {
            self.outline_scale = 1.0 + rs.flux_outline.max(0.0) * self.flux_env;
        }

        // Normalised band energies (idea 4 / 11)
        if rs.band_sculpt || rs.band_colors {
            for b in 0..4 {
                self.band_n[b] = self.band_range[b].update(bands[b], dt, 12.0, 0.05) * activity;
            }
        }

        // Chroma (idea 10)
        if rs.chroma_rotate || rs.chroma_bars {
            let k = 1.0 - (-dt / 0.12).exp();
            for pc in 0..12 { self.chroma_s[pc] += (chroma[pc] - self.chroma_s[pc]) * k; }
            let (mut mn, mut mx, mut arg) = (f32::MAX, f32::MIN, 0usize);
            for pc in 0..12 {
                let v = self.chroma_s[pc];
                if v < mn { mn = v; }
                if v > mx { mx = v; arg = pc; }
            }
            self.chroma_min = mn;
            self.chroma_span = mx - mn;
            if self.chroma_span > 0.04
                && arg != self.key_pc
                && self.chroma_s[arg] > self.chroma_s[self.key_pc] + 0.06
            {
                self.key_pc = arg;
            }
            let target = self.key_pc as f32 * 30.0;
            let delta = (target - self.rot_deg + 540.0).rem_euclid(360.0) - 180.0;
            let step = rs.chroma_rotate_speed.max(1.0) * dt;
            self.rot_deg = (self.rot_deg + delta.clamp(-step, step)).rem_euclid(360.0);
        }
        self.hue_deg = if rs.chroma_rotate { self.rot_deg * rs.chroma_rotate_amount.clamp(0.0, 1.0) } else { 0.0 };

        // Dynamics colour transform
        self.dyn_on = rs.dynamics_color;
        if self.dyn_on {
            let k = self.dynamics.intensity.clamp(0.0, 1.0);
            self.dyn_sat = lerp(rs.dyn_low_saturation, rs.dyn_high_saturation, k);
            self.dyn_val = lerp(rs.dyn_low_brightness, rs.dyn_high_brightness, k);
        }

        // Dominant pitch tracking (idea 12)
        let mut pos1: Option<f32> = None;
        let mut pos2: Option<f32> = None;
        let mut pos3: Option<f32> = None;
        if rs.pitch_highlight {
            let valid = pitch_hz >= 100.0 && activity > 0.3 && self.hz_to_pos(pitch_hz).is_some();
            if valid {
                let target = pitch_hz.ln();
                if self.pitch_w < 0.05 || (target - self.pitch_ln).abs() > 0.5 {
                    self.pitch_ln = target;
                } else {
                    let k = 1.0 - (-dt / (rs.pitch_glide_ms / 1000.0).max(0.001)).exp();
                    self.pitch_ln += (target - self.pitch_ln) * k;
                }
                self.since_pitch = 0.0;
            } else {
                self.since_pitch += dt;
            }
            let want = if self.since_pitch * 1000.0 < rs.pitch_hold_ms { 1.0 } else { 0.0 };
            let tau = if want > self.pitch_w { 0.04 } else { 0.12 };
            self.pitch_w += (want - self.pitch_w) * (1.0 - (-dt / tau).exp());
            if self.pitch_w > 0.01 {
                let hz = self.pitch_ln.exp();
                pos1 = self.hz_to_pos(hz);
                if rs.pitch_harmonics {
                    pos2 = self.hz_to_pos(hz * 2.0);
                    pos3 = self.hz_to_pos(hz * 3.0);
                }
            }
            self.pitch_w_public = self.pitch_w;
        }

        // Waveform per bar (idea 13)
        if rs.wave_wiggle {
            if let Some(fr) = f { self.wave_per_bar(&fr.waveform, n); } else { reset!(self.wave_bar, 0.0f32); }
        }

        // Shared per-frame constants
        let denom = (n.max(2) - 1) as f32;
        let k_wave = rs.pw_count * TAU / n as f32;
        let s_phase = phase * TAU;
        let sweep_w = rs.sweep_width.clamp(0.01, 0.5);
        let head_u = match rs.sweep_mode {
            SweepMode::Forward => phase,
            SweepMode::Reverse => 1.0 - phase,
            SweepMode::PingPong => 1.0 - (2.0 * phase - 1.0).abs(),
        };
        let sweep_head = -sweep_w + head_u * (1.0 + 2.0 * sweep_w);
        let windup_k = if rs.windup && beat_ok {
            let period = 60.0 / bpm;
            let wnd = rs.windup_window.clamp(0.05, 0.9);
            (1.0 - (beat_in / (period * wnd)).clamp(0.0, 1.0)) * activity
        } else { 0.0 };
        let drum_amt = [rs.spike_kick, rs.spike_snare, rs.spike_hat];
        let shimmer_seed = ((wall_time * 30.0) as u32).wrapping_mul(0x9E37_79B1);
        let spike_on = rs.onset_spikes;
        let band_rgb: [[f32; 3]; 4] = [
            rgb01(rs.band_color_list[0]), rgb01(rs.band_color_list[1]),
            rgb01(rs.band_color_list[2]), rgb01(rs.band_color_list[3]),
        ];
        let flash_rgb: [[f32; 3]; 3] = [
            rgb01(rs.flash_colors[0]), rgb01(rs.flash_colors[1]), rgb01(rs.flash_colors[2]),
        ];
        let sweep_rgb = rgb01(rs.sweep_color);
        let pitch_rgb = rgb01(rs.pitch_color);
        let tonal = self.chroma_span > 0.04;

        for idx in 0..n {
            let hz = self.bar_hz.get(idx).copied().unwrap_or(1000.0);
            let mut hm = 1.0f32;
            let mut pm = 1.0f32;
            let mut shift = 0.0f32;
            let mut wm = 1.0f32;
            let mut dim = 1.0f32;
            let mut ov = [0.0f32; 4];
            let mut add = [0.0f32; 3];

            // 1 — per-band onset spikes
            if spike_on {
                let mut s = 0.0;
                for d in 0..3 { s += drum_amt[d] * self.spike_env[d] * self.w_drum[d][idx]; }
                hm += s;
            }

            // 2 — beat envelope pump (predicted beats already arrive as 0.7)
            if rs.beat_pump {
                hm *= 1.0 + rs.pump_height * beat_env;
                wm *= 1.0 + rs.pump_width * beat_env;
                pm *= 1.0 + rs.pump_peak * beat_env;
            }

            // 3 — pre-beat wind-up
            if windup_k > 0.0 {
                hm *= 1.0 - rs.windup_depth * windup_k;
                over(&mut ov, [1.0, 1.0, 1.0], rs.windup_brighten * windup_k);
            }

            // 4 — band sculpting
            if rs.band_sculpt {
                let bn = self.band_interp(hz);
                hm *= (1.0 + rs.sculpt_bass * self.band_n[0]) * (1.0 + rs.sculpt_region * bn);
                pm *= 1.0 + rs.sculpt_peak * bn;
            }

            // 5 — transient snap & flux shimmer
            if rs.transient_snap {
                hm *= 1.0 + rs.snap_amount * self.snap_env;
                if rs.flux_shimmer > 0.0 {
                    let h = hash_u32((idx as u32) ^ shimmer_seed);
                    let r = (h as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    hm *= 1.0 + rs.flux_shimmer * self.flux_env * r;
                }
            }

            // 7 — phase sweep
            if rs.phase_sweep && beat_ok {
                let u = idx as f32 / denom;
                let d = ((u - sweep_head).abs() / sweep_w).min(1.0);
                let prof = (1.0 - smoothstep(0.0, 1.0, d)) * activity;
                over(&mut ov, sweep_rgb, rs.sweep_strength * prof);
                hm *= 1.0 + rs.sweep_lift * prof;
            }

            // 8 — phase wave lean / breathing
            if rs.phase_wave && beat_ok {
                let s = (s_phase + idx as f32 * k_wave).sin();
                shift += s * rs.pw_sway_px * activity;
                hm *= 1.0 + rs.pw_height * s * activity;
                dim *= 1.0 - rs.pw_dim * phase * activity;
            }

            // 9 — per-band onset flash overlay
            if rs.onset_flash {
                for d in 0..3 {
                    let a = self.flash_env[d] * self.w_drum[d][idx] * rs.flash_strength;
                    if a > 0.002 {
                        let rgb = match rs.flash_mode {
                            FlashMode::Brighten => [1.0, 1.0, 1.0],
                            FlashMode::Color => flash_rgb[d],
                        };
                        over(&mut ov, rgb, a);
                    }
                }
            }

            // 10 — per-bar pitch-class lighting
            if rs.chroma_bars && tonal {
                let pc = self.bar_pc.get(idx).copied().unwrap_or(-1);
                if pc >= 0 {
                    let v = ((self.chroma_s[pc as usize] - self.chroma_min) / self.chroma_span.max(1e-3))
                        .clamp(0.0, 1.0)
                        .powf(rs.chroma_contrast.clamp(0.5, 6.0)) * activity;
                    over(&mut ov, hsv_to_rgb(pc as f32 / 12.0, 0.85, 1.0), rs.chroma_tint * v);
                    over(&mut ov, [1.0, 1.0, 1.0], rs.chroma_brighten * v);
                }
            }

            // 11 — live 4-band colour blending
            if rs.band_colors {
                let sharp = rs.band_color_sharpness.clamp(0.5, 4.0);
                let sp = rs.band_color_spatial.clamp(0.0, 1.0);
                match rs.band_color_mode {
                    BandColorMode::Blend => {
                        let (mut wsum, mut wmax) = (0.0f32, 0.0f32);
                        let mut col = [0.0f32; 3];
                        for b in 0..4 {
                            let w = self.band_n[b].powf(sharp) * lerp(1.0, Self::band_affinity(b, hz), sp);
                            wsum += w;
                            wmax = wmax.max(w);
                            for k in 0..3 { col[k] += band_rgb[b][k] * w; }
                        }
                        if wsum > 1e-4 {
                            for k in 0..3 { col[k] /= wsum; }
                            over(&mut ov, col, rs.band_color_strength * wmax.clamp(0.0, 1.0));
                        }
                    }
                    BandColorMode::WarmCool => {
                        let warm = self.band_n[0].powf(sharp) * lerp(1.0, Self::band_affinity(0, hz), sp);
                        let cool = self.band_n[3].powf(sharp) * lerp(1.0, Self::band_affinity(3, hz), sp);
                        for k in 0..3 {
                            add[k] += (band_rgb[0][k] * warm + band_rgb[3][k] * cool)
                                * rs.band_color_strength * 0.35;
                        }
                    }
                }
            }

            // 12 — dominant pitch highlight (+ harmonics)
            if let Some(p1) = pos1 {
                let w = rs.pitch_width_bars.max(0.5);
                let tri = |p: f32| (1.0 - (idx as f32 - p).abs() / w).clamp(0.0, 1.0);
                let mut weight = tri(p1);
                if let Some(p2) = pos2 { weight += rs.pitch_harmonic_strength * tri(p2); }
                if let Some(p3) = pos3 { weight += rs.pitch_harmonic_strength * 0.6 * tri(p3); }
                let weight = (weight * self.pitch_w).clamp(0.0, 1.0);
                self.pitch_bar[idx] = weight;
                over(&mut ov, pitch_rgb, rs.pitch_strength * weight);
            }

            // 13 — waveform tip wiggle / x-vibration
            if rs.wave_wiggle {
                let s = self.wave_bar.get(idx).copied().unwrap_or(0.0);
                hm *= 1.0 + rs.wiggle_amount * s;
                shift += rs.wiggle_x_px * s;
            }

            self.height_mul[idx] = hm.max(0.0);
            self.peak_mul[idx] = pm.max(0.1);
            self.x_shift[idx] = shift;
            self.width_mul[idx] = wm.max(0.1);
            self.dim[idx] = dim.max(0.0);
            self.overlay[idx] = ov;
            self.add[idx] = add;
        }

        // 14 — ghost line source
        if rs.ghost_line {
            if let Some(fr) = f {
                match rs.ghost_source {
                    GhostSource::Aligned => {
                        self.ghost.extend((0..n).map(|i| fr.instant.get(i).copied().unwrap_or(0.0)));
                    }
                    GhostSource::RawSpectrum => {
                        let l = fr.spectrum_row.len();
                        if l >= 2 {
                            for i in 0..n {
                                let pos = i as f32 / denom * (l - 1) as f32;
                                let a = (pos.floor() as usize).min(l - 1);
                                let b = (a + 1).min(l - 1);
                                self.ghost.push(lerp(fr.spectrum_row[a], fr.spectrum_row[b], pos - a as f32));
                            }
                        }
                    }
                }
            }
        }

        self.color_active = (rs.windup && rs.windup_brighten > 0.0)
            || rs.phase_sweep || (rs.phase_wave && rs.pw_dim > 0.0) || rs.onset_flash
            || rs.chroma_rotate || rs.chroma_bars || rs.band_colors
            || rs.pitch_highlight || rs.dynamics_color;
    }

    fn wave_per_bar(&mut self, wf: &[f32], n: usize) {
        self.wave_bar.clear();
        self.wave_bar.resize(n, 0.0);
        if wf.is_empty() || n == 0 { return; }
        let peak = wf.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let gate = smoothstep(0.003, 0.08, peak);
        if gate <= 0.0 || peak <= 1e-6 { return; }
        let len = wf.len();
        for i in 0..n {
            let a = i * len / n;
            let b = ((i + 1) * len / n).max(a + 1).min(len);
            let s = wf[(a + b) / 2];
            self.wave_bar[i] = (s / peak).clamp(-1.0, 1.0) * gate;
        }
    }

    // ── accessors used while drawing ───────────────────────────
    #[inline]
    pub fn apply_len(&self, idx: usize, base: f32, cap: f32) -> f32 {
        if !self.active || base <= 0.0 { return base; }
        let m = self.height_mul.get(idx).copied().unwrap_or(1.0);
        (base * m).clamp(0.0, cap + FX_HEADROOM_PX)
    }
    #[inline]
    pub fn shift(&self, idx: usize) -> f32 {
        if self.active { self.x_shift.get(idx).copied().unwrap_or(0.0) } else { 0.0 }
    }
    #[inline]
    pub fn width_mul(&self, idx: usize) -> f32 {
        if self.active { self.width_mul.get(idx).copied().unwrap_or(1.0) } else { 1.0 }
    }
    #[inline]
    pub fn peak_scale(&self, idx: usize) -> f32 {
        if self.active { self.peak_mul.get(idx).copied().unwrap_or(1.0) } else { 1.0 }
    }

    /// Applies every colour effect to a bar colour (alpha is preserved).
    pub fn tint(&self, c: egui::Color32, idx: usize) -> egui::Color32 {
        if !self.color_active || idx >= self.overlay.len() { return c; }
        let [r8, g8, b8, a8] = c.to_srgba_unmultiplied();
        let (mut r, mut g, mut b) = (r8 as f32 / 255.0, g8 as f32 / 255.0, b8 as f32 / 255.0);

        if self.dyn_on {
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            r = (l + (r - l) * self.dyn_sat) * self.dyn_val;
            g = (l + (g - l) * self.dyn_sat) * self.dyn_val;
            b = (l + (b - l) * self.dyn_sat) * self.dyn_val;
        }
        if self.hue_deg.abs() > 0.1 {
            // Hue rotation in YIQ space keeps perceived brightness stable.
            let (s, co) = self.hue_deg.to_radians().sin_cos();
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let i = 0.596 * r - 0.274 * g - 0.322 * b;
            let q = 0.211 * r - 0.523 * g + 0.312 * b;
            let i2 = i * co - q * s;
            let q2 = i * s + q * co;
            r = y + 0.956 * i2 + 0.621 * q2;
            g = y - 0.272 * i2 - 0.647 * q2;
            b = y - 1.106 * i2 + 1.703 * q2;
        }
        let d = self.dim[idx];
        r *= d; g *= d; b *= d;
        let o = self.overlay[idx];
        if o[3] > 0.0 {
            r = lerp(r, o[0], o[3]);
            g = lerp(g, o[1], o[3]);
            b = lerp(b, o[2], o[3]);
        }
        let ad = self.add[idx];
        r += ad[0]; g += ad[1]; b += ad[2];

        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        egui::Color32::from_rgba_unmultiplied(q(r), q(g), q(b), a8)
    }
}

// ───────────────────────────────────────────────────────────────
// Drawing helpers on VisualizerApp
// ───────────────────────────────────────────────────────────────

fn scale_alpha(c: egui::Color32, k: f32) -> egui::Color32 {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    egui::Color32::from_rgba_unmultiplied(r, g, b, (a as f32 * k.clamp(0.0, 1.0)) as u8)
}

impl VisualizerApp {
    /// Colour effects for the bar drawn at *visual* index `i_visual`.
    #[inline]
    pub(crate) fn fx_color(&self, c: egui::Color32, i_visual: usize) -> egui::Color32 {
        if !self.fx.color_active { return c; }
        let base = self.current_bars.len().max(1);
        let idx = frequency_index(i_visual, base, self.config.frequency_direction).min(base - 1);
        self.fx.tint(c, idx)
    }

    /// Same, but blended between two neighbouring bars (Smooth mode segments).
    #[inline]
    pub(crate) fn fx_color_lerp(&self, c: egui::Color32, ia: usize, ib: usize, t: f32) -> egui::Color32 {
        if !self.fx.color_active { return c; }
        lerp_color(self.fx_color(c, ia), self.fx_color(c, ib), t)
    }

    /// Bar length in pixels with all geometry effects applied.
    #[inline]
    pub(crate) fn bar_len_px(&self, idx: usize, rect_len: f32, max_h: f32) -> f32 {
        let cap = if max_h > 0.0 { max_h } else { rect_len };
        let base = (self.current_bars[idx] * rect_len).min(cap);
        self.fx.apply_len(idx, base, cap)
    }

    /// Peak-cap position in pixels (kept at or above the bar it belongs to).
    #[inline]
    pub(crate) fn peak_len_px(&self, idx: usize, rect_len: f32, max_h: f32, bar_len: f32) -> f32 {
        let cap = if max_h > 0.0 { max_h } else { rect_len };
        let base = (self.current_peaks[idx] * rect_len).min(cap);
        let l = self.fx.apply_len(idx, base, cap);
        if self.fx.active { l.max(bar_len) } else { l }
    }

    /// Ghost line + dominant-pitch markers, drawn on top of the bars.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_fx_overlays(
        &self,
        painter: &egui::Painter,
        dir: VisualizerDirection,
        start_pos: f32,
        anchor: f32,
        rect_len: f32,
        bar_width: f32,
        stride: f32,
        visual_num_bars: usize,
        base_num_bars: usize,
        freq_dir: FrequencyDirection,
        max_h: f32,
    ) {
        let rs = &self.config.reactive;
        if !self.fx.active || base_num_bars == 0 { return; }
        let cap = if max_h > 0.0 { max_h } else { rect_len };
        let is_smooth = self.config.bar_interpolation == BarInterpolation::Smooth;

        // 14 — ghost line
        if rs.ghost_line && self.fx.ghost.len() == base_num_bars && visual_num_bars > 1 {
            let mut pts = Vec::with_capacity(visual_num_bars);
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let v = self.fx.ghost[idx].clamp(0.0, 1.0) * rs.ghost_gain.max(0.0);
                let len = (v * rect_len).min(cap + FX_HEADROOM_PX);
                let perp = start_pos + i as f32 * stride + bar_width * 0.5 + self.fx.shift(idx);
                pts.push(bar_vertex(dir, perp, anchor, len, 1.0));
            }
            let c = rs.ghost_color;
            let col = self.apply_vis(egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], rs.ghost_opacity));
            painter.add(egui::Shape::line(pts, egui::Stroke::new(rs.ghost_thickness.max(0.5), col)));
        }

        // 12 — dominant pitch cap / outline
        if rs.pitch_highlight && self.fx.pitch_w_public > 0.02 && self.fx.pitch_bar.len() == base_num_bars {
            let c = rs.pitch_color;
            let accent_col = self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2]));
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let w = self.fx.pitch_bar[idx];
                if w < 0.15 { continue; }
                let len = self.bar_len_px(idx, rect_len, max_h);
                if len <= 0.5 { continue; }
                let bw = bar_width * self.fx.width_mul(idx);
                let perp_start = start_pos + i as f32 * stride + self.fx.shift(idx)
                    - (bw - bar_width) * 0.5;
                let col = scale_alpha(accent_col, w);
                if rs.pitch_cap_px > 0.0 {
                    draw_peak_hold(
                        painter, dir, anchor, perp_start + bw * 0.5, len, bw, col,
                        is_smooth, rs.pitch_cap_px / 3.0,
                    );
                }
                if rs.pitch_outline {
                    let r = self.bar_rect(dir, perp_start, bw, anchor, len);
                    painter.rect_stroke(r, 0.0, egui::Stroke::new(1.5, col));
                }
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────
// Settings tab
// ───────────────────────────────────────────────────────────────

fn sub_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new(text).strong().size(14.0));
}

fn note(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(12.0).color(egui::Color32::from_rgb(230, 170, 90)));
}

impl SettingsState {
    pub(crate) fn tab_reactive(&mut self, ui: &mut egui::Ui) {
        let show_help = self.config.show_help_text;
        let bpm_known = self.last_bpm >= 20.0;
        let chroma_on = self.config.audio.chroma_enabled;
        let intensity = self.fx_intensity;
        let decay_mode_gravity =
            self.config.audio.decay_mode == visualizer_core::config::DecayMode::Gravity;
        let mut enable_chroma = false;
        let mut reset_memory = false;
        let d = ReactiveSettings::default();
        let r = &mut self.config.reactive;

        section_header(ui, "Reactive Effects");
        ui.horizontal(|ui| {
            ui.checkbox(&mut r.enabled, "Enable reactive effects");
            if ui.button("Turn all effects off").clicked() {
                r.onset_spikes = false; r.beat_pump = false; r.windup = false;
                r.band_sculpt = false; r.transient_snap = false; r.beat_clock_colors = false;
                r.phase_sweep = false; r.phase_wave = false; r.onset_flash = false;
                r.chroma_rotate = false; r.chroma_bars = false; r.band_colors = false;
                r.pitch_highlight = false; r.wave_wiggle = false; r.ghost_line = false;
                r.dynamics_color = false;
            }
            if ui.button("Reset all to defaults").clicked() {
                *r = ReactiveSettings::default();
            }
        });
        help_text(
            ui,
            show_help,
            "Visual-only effects driven by the audio analysis. Each one is off by default and \
             independent; combine as many as you like. Height effects may exceed the max bar \
             height by up to 24 px.",
        );
        if !bpm_known {
            note(ui, "No tempo detected yet — phase / beat-in effects (3, 7, 8) stay idle until a BPM is found.");
        }

        // ══ A ═════════════════════════════════════════════════
        section_header(ui, "A · Rhythm & Transient Geometry");

        sub_header(ui, "1 · Per-Band Onset Spikes");
        checkbox_reset(ui, &mut r.onset_spikes, "Enable", false);
        if r.onset_spikes {
            slider_f32(ui, &mut r.spike_kick, 0.0..=1.0, "Kick boost (bass bars)", d.spike_kick);
            slider_f32(ui, &mut r.spike_snare, 0.0..=1.0, "Snare boost (mid bars)", d.spike_snare);
            slider_f32(ui, &mut r.spike_hat, 0.0..=1.0, "Hat boost (treble bars)", d.spike_hat);
            slider_f32(ui, &mut r.spike_decay_ms, 30.0..=600.0, "Decay (ms)", d.spike_decay_ms);
            slider_f32(ui, &mut r.spike_softness_oct, 0.05..=1.5, "Region softness (octaves)", d.spike_softness_oct);
            help_text(ui, show_help,
                "Kick lifts 20–150 Hz bars, snare 150 Hz–2 kHz, hat 6–12 kHz. Each drum fires on its own, \
                 so you get polyrhythmic movement instead of one uniform jump.");
        }

        sub_header(ui, "2 · Beat Envelope Pump");
        checkbox_reset(ui, &mut r.beat_pump, "Enable", false);
        if r.beat_pump {
            slider_f32(ui, &mut r.pump_height, 0.0..=0.5, "Height pump", d.pump_height);
            slider_f32(ui, &mut r.pump_width, 0.0..=0.5, "Width swell", d.pump_width);
            slider_f32(ui, &mut r.pump_peak, 0.0..=3.0, "Peak-cap swell", d.pump_peak);
            help_text(ui, show_help,
                "Driven by beat_envelope: real beats hit at full strength, BPM-lock predicted beats at 0.7×, \
                 so lock drift is visible. Peak-cap swell needs Peak Hold > 0 (Audio → Ballistics). \
                 Width swell applies to the Step shape only and can overlap neighbours slightly when Bar Gap is small.");
        }

        sub_header(ui, "3 · Pre-Beat Wind-Up");
        checkbox_reset(ui, &mut r.windup, "Enable", false);
        if r.windup {
            slider_f32(ui, &mut r.windup_depth, 0.0..=0.3, "Compression depth", d.windup_depth);
            slider_f32(ui, &mut r.windup_window, 0.05..=0.9, "Wind-up window (fraction of beat)", d.windup_window);
            slider_f32(ui, &mut r.windup_brighten, 0.0..=0.6, "Brightness ramp", d.windup_brighten);
            help_text(ui, show_help,
                "Bars squash slightly during the last part of each beat, then release as it lands — the only \
                 effect that reacts ahead of the beat. Works best with BPM Lock enabled.");
        }

        sub_header(ui, "4 · Sub-Bass / Band Sculpting");
        checkbox_reset(ui, &mut r.band_sculpt, "Enable", false);
        if r.band_sculpt {
            slider_f32(ui, &mut r.sculpt_bass, 0.0..=0.5, "Global bass weight", d.sculpt_bass);
            slider_f32(ui, &mut r.sculpt_region, 0.0..=0.5, "Per-region lift", d.sculpt_region);
            slider_f32(ui, &mut r.sculpt_peak, 0.0..=3.0, "Peak-cap size by band energy", d.sculpt_peak);
            help_text(ui, show_help,
                "Slow macro texture from the 4 band levels, each normalised to its own recent range so it \
                 reads as 'this band is loud right now' rather than a constant offset.");
        }

        sub_header(ui, "5 · Transient Snap & Flux Shimmer");
        checkbox_reset(ui, &mut r.transient_snap, "Enable", false);
        if r.transient_snap {
            slider_f32(ui, &mut r.snap_amount, 0.0..=0.15, "Snap height", d.snap_amount);
            slider_f32(ui, &mut r.snap_threshold, 0.1..=1.0, "Snap threshold (sample peak)", d.snap_threshold);
            slider_f32(ui, &mut r.flux_shimmer, 0.0..=0.1, "Flux shimmer", d.flux_shimmer);
            slider_f32(ui, &mut r.flux_outline, 0.0..=3.0, "Outline thickness by flux", d.flux_outline);
            help_text(ui, show_help,
                "Keep these small so bar heights still read accurately. Outline thickness only shows when \
                 outlines are enabled (Visualizer tab).");
        }

        // ══ B ═════════════════════════════════════════════════
        section_header(ui, "B · Phase-Locked & Tempo-Synced Motion");

        sub_header(ui, "6 · Beat-Clock Animated Colors");
        checkbox_reset(ui, &mut r.beat_clock_colors, "Enable", false);
        if r.beat_clock_colors {
            slider_f32(ui, &mut r.beat_clock_beats_per_cycle, 0.5..=32.0, "Beats per color cycle", d.beat_clock_beats_per_cycle);
            help_text(ui, show_help,
                "Replaces the wall clock used by Colors → Animated with the song's tempo. All 30 motion \
                 shapes lock to the beat and follow Half/Double-Time. The Motion Speed multiplier still applies. \
                 Without a detected tempo it runs at normal speed.");
        }

        sub_header(ui, "7 · Phase Sweep Highlight");
        checkbox_reset(ui, &mut r.phase_sweep, "Enable", false);
        if r.phase_sweep {
            combo_reset(ui, &mut r.sweep_mode, SweepMode::default(), "Direction", "fx_sweep_mode",
                        170.0, &SweepMode::all(), SweepMode::display_name);
            slider_f32(ui, &mut r.sweep_width, 0.02..=0.5, "Width (fraction of spectrum)", d.sweep_width);
            slider_f32(ui, &mut r.sweep_strength, 0.0..=1.0, "Strength", d.sweep_strength);
            slider_f32(ui, &mut r.sweep_lift, 0.0..=0.3, "Height lift", d.sweep_lift);
            color_edit_reset(ui, &mut r.sweep_color, "Shine color", d.sweep_color);
            help_text(ui, show_help,
                "A shine walks across the spectrum once per beat (a metronome inside the bars). It follows \
                 frequency order, so in Mirrored layout it travels outward from the centre. Respects Latency Offset.");
        }

        sub_header(ui, "8 · Phase Wave Lean / Breathing");
        checkbox_reset(ui, &mut r.phase_wave, "Enable", false);
        if r.phase_wave {
            slider_f32(ui, &mut r.pw_sway_px, 0.0..=20.0, "Horizontal sway (px)", d.pw_sway_px);
            slider_f32(ui, &mut r.pw_height, 0.0..=0.3, "Height wave", d.pw_height);
            slider_f32(ui, &mut r.pw_count, 0.25..=8.0, "Waves across spectrum", d.pw_count);
            slider_f32(ui, &mut r.pw_dim, 0.0..=0.7, "Brightness decay per beat", d.pw_dim);
            help_text(ui, show_help,
                "Sinusoidal sway / breathing locked to the beat phase. Brightness decay gives a clean emphasis \
                 at the start of every beat without flashing.");
        }

        // ══ C ═════════════════════════════════════════════════
        section_header(ui, "C · Spectral & Harmonic Coloring");

        sub_header(ui, "9 · Per-Band Onset Flash");
        checkbox_reset(ui, &mut r.onset_flash, "Enable", false);
        if r.onset_flash {
            combo_reset(ui, &mut r.flash_mode, FlashMode::default(), "Flash style", "fx_flash_mode",
                        230.0, &FlashMode::all(), FlashMode::display_name);
            slider_f32(ui, &mut r.flash_strength, 0.0..=1.0, "Strength", d.flash_strength);
            slider_f32(ui, &mut r.flash_decay_ms, 40.0..=1000.0, "Decay (ms)", d.flash_decay_ms);
            if r.flash_mode == FlashMode::Color {
                color_edit_reset(ui, &mut r.flash_colors[0], "Kick color", d.flash_colors[0]);
                color_edit_reset(ui, &mut r.flash_colors[1], "Snare color", d.flash_colors[1]);
                color_edit_reset(ui, &mut r.flash_colors[2], "Hat color", d.flash_colors[2]);
            }
            help_text(ui, show_help, "Brightness/colour only — bar geometry is untouched.");
        }

        sub_header(ui, "10 · Chroma Wheel & Chord Lighting");
        if !chroma_on {
            ui.horizontal(|ui| {
                note(ui, "Chroma analysis is off (Audio → Chroma).");
                if ui.small_button("Turn it on").clicked() { enable_chroma = true; }
            });
        }
        checkbox_reset(ui, &mut r.chroma_rotate, "Palette rotation (follows key / chord)", false);
        if r.chroma_rotate {
            slider_f32(ui, &mut r.chroma_rotate_amount, 0.0..=1.0, "Rotation amount", d.chroma_rotate_amount);
            slider_f32(ui, &mut r.chroma_rotate_speed, 10.0..=720.0, "Rotation speed (°/s)", d.chroma_rotate_speed);
            help_text(ui, show_help,
                "The palette's hue rotates by 30° per pitch class of the strongest note. It glides along the \
                 shortest path instead of jumping when the chord changes.");
        }
        checkbox_reset(ui, &mut r.chroma_bars, "Per-bar pitch-class lighting", false);
        if r.chroma_bars {
            slider_f32(ui, &mut r.chroma_tint, 0.0..=1.0, "Pitch-class hue tint", d.chroma_tint);
            slider_f32(ui, &mut r.chroma_brighten, 0.0..=1.0, "Brighten active notes", d.chroma_brighten);
            slider_f32(ui, &mut r.chroma_contrast, 0.5..=6.0, "Contrast", d.chroma_contrast);
            help_text(ui, show_help,
                "Bars whose pitch class is sounding light up (only bars between 200 Hz and 4 kHz, the range \
                 chroma analyses). Color only — spectral accuracy stays intact.");
        }

        sub_header(ui, "11 · Live 4-Band Color Blending");
        checkbox_reset(ui, &mut r.band_colors, "Enable", false);
        if r.band_colors {
            combo_reset(ui, &mut r.band_color_mode, BandColorMode::default(), "Mode", "fx_band_color_mode",
                        230.0, &BandColorMode::all(), BandColorMode::display_name);
            slider_f32(ui, &mut r.band_color_strength, 0.0..=1.0, "Strength", d.band_color_strength);
            slider_f32(ui, &mut r.band_color_sharpness, 0.5..=4.0, "Sharpness", d.band_color_sharpness);
            slider_f32(ui, &mut r.band_color_spatial, 0.0..=1.0, "Spatial (per-region) amount", d.band_color_spatial);
            let names = ["Sub (20–100 Hz)", "Low (100–500 Hz)", "Mid (0.5–2 kHz)", "High (2–8 kHz)"];
            for b in 0..4 {
                color_edit_reset(ui, &mut r.band_color_list[b], names[b], d.band_color_list[b]);
            }
            help_text(ui, show_help,
                "Blend: the four colors are mixed by how loud each band is. Warm/Cool: additive warm tint from \
                 Sub and cool tint from High. Spatial = 0 tints all bars equally; higher values tint each bar by \
                 its own frequency region. No beat detection needed.");
        }

        // ══ D ═════════════════════════════════════════════════
        section_header(ui, "D · Pitch & Raw-Signal Detail");

        sub_header(ui, "12 · Dominant Pitch Highlight & Harmonics");
        if !chroma_on {
            ui.horizontal(|ui| {
                note(ui, "Pitch tracking needs chroma analysis (Audio → Chroma).");
                if ui.small_button("Turn it on").clicked() { enable_chroma = true; }
            });
        }
        checkbox_reset(ui, &mut r.pitch_highlight, "Enable", false);
        if r.pitch_highlight {
            color_edit_reset(ui, &mut r.pitch_color, "Accent color", d.pitch_color);
            slider_f32(ui, &mut r.pitch_strength, 0.0..=1.0, "Color strength", d.pitch_strength);
            slider_f32(ui, &mut r.pitch_width_bars, 0.5..=8.0, "Highlight width (bars)", d.pitch_width_bars);
            slider_f32(ui, &mut r.pitch_cap_px, 0.0..=12.0, "Accent cap thickness (px)", d.pitch_cap_px);
            checkbox_reset(ui, &mut r.pitch_outline, "Outline highlighted bars", false);
            checkbox_reset(ui, &mut r.pitch_harmonics, "Emphasise 2× and 3× harmonics", false);
            if r.pitch_harmonics {
                slider_f32(ui, &mut r.pitch_harmonic_strength, 0.0..=1.0, "Harmonic strength", d.pitch_harmonic_strength);
            }
            slider_f32(ui, &mut r.pitch_hold_ms, 0.0..=1000.0, "Hold when pitch drops out (ms)", d.pitch_hold_ms);
            slider_f32(ui, &mut r.pitch_glide_ms, 5.0..=300.0, "Glide time (ms)", d.pitch_glide_ms);
            help_text(ui, show_help,
                "Marks the bar nearest the loudest tone (200 Hz–4 kHz) — handy for tracking a lead voice or solo \
                 instrument. The accent cap is drawn independently of Peak Hold.");
        }

        sub_header(ui, "13 · Waveform Tip Wiggle");
        checkbox_reset(ui, &mut r.wave_wiggle, "Enable", false);
        if r.wave_wiggle {
            slider_f32(ui, &mut r.wiggle_amount, 0.0..=0.15, "Tip wiggle (fraction of bar height)", d.wiggle_amount);
            slider_f32(ui, &mut r.wiggle_x_px, 0.0..=6.0, "Horizontal micro-vibration (px)", d.wiggle_x_px);
            if decay_mode_gravity {
                note(ui, "Decay Mode is Gravity — the wiggle reads best with Linear or Exponential decay.");
            }
            help_text(ui, show_help,
                "The raw waveform (last 1024 samples) is laid along the bars so the tips visibly vibrate with \
                 the audio signal. Works in Step and Smooth shapes; Smooth looks best.");
        }

        sub_header(ui, "14 · Raw Spectrum Ghost Line");
        checkbox_reset(ui, &mut r.ghost_line, "Enable", false);
        if r.ghost_line {
            combo_reset(ui, &mut r.ghost_source, GhostSource::default(), "Source", "fx_ghost_source",
                        250.0, &GhostSource::all(), GhostSource::display_name);
            slider_f32(ui, &mut r.ghost_thickness, 0.5..=6.0, "Thickness (px)", d.ghost_thickness);
            slider_u8(ui, &mut r.ghost_opacity, "Opacity", d.ghost_opacity);
            slider_f32(ui, &mut r.ghost_gain, 0.25..=2.0, "Height gain", d.ghost_gain);
            color_edit_reset(ui, &mut r.ghost_color, "Line color", d.ghost_color);
            help_text(ui, show_help,
                "A thin line showing the spectrum before attack/decay smoothing, so you see transients before \
                 the bars catch up. 'Pre-ballistics bars' shares the bars' exact scale and frequency mapping. \
                 'Raw spectrum_row' uses a fixed 91 dB scale on a log axis, so it will not line up with the \
                 bars' height or with Mel/Bark/Linear scaling.");
        }

        // ══ Dynamics ══════════════════════════════════════════
        section_header(ui, "Dynamics Color (track-relative intensity)");
        checkbox_reset(ui, &mut r.dynamics_color, "Enable", false);
        if r.dynamics_color {
            ui.horizontal(|ui| {
                ui.label("Intensity now:");
                ui.add(egui::ProgressBar::new(intensity.clamp(0.0, 1.0)).desired_width(220.0).show_percentage());
                if ui.small_button("Forget this track").clicked() { reset_memory = true; }
            });
            sub_header(ui, "Look");
            slider_f32(ui, &mut r.dyn_low_saturation, 0.0..=1.5, "Saturation when calm", d.dyn_low_saturation);
            slider_f32(ui, &mut r.dyn_high_saturation, 0.5..=2.5, "Saturation when intense", d.dyn_high_saturation);
            slider_f32(ui, &mut r.dyn_low_brightness, 0.3..=1.2, "Brightness when calm", d.dyn_low_brightness);
            slider_f32(ui, &mut r.dyn_high_brightness, 0.7..=1.6, "Brightness when intense", d.dyn_high_brightness);
            sub_header(ui, "Analysis");
            slider_f32(ui, &mut r.dyn_contrast, 0.5..=3.0, "Contrast (raise for subtle music)", d.dyn_contrast);
            slider_f32(ui, &mut r.dyn_memory_s, 10.0..=300.0, "Track memory (s)", d.dyn_memory_s);
            slider_f32(ui, &mut r.dyn_attack_ms, 20.0..=1000.0, "Rise time (ms)", d.dyn_attack_ms);
            slider_f32(ui, &mut r.dyn_release_ms, 100.0..=6000.0, "Fall time (ms)", d.dyn_release_ms);
            slider_f32(ui, &mut r.dyn_w_loudness, 0.0..=2.0, "Weight: loudness vs. rest of track", d.dyn_w_loudness);
            slider_f32(ui, &mut r.dyn_w_attack, 0.0..=2.0, "Weight: attack strength", d.dyn_w_attack);
            slider_f32(ui, &mut r.dyn_w_brightness, 0.0..=2.0, "Weight: spectral brightness", d.dyn_w_brightness);
            help_text(ui, show_help,
                "Measures how hard the music is being played relative to this track's own history. Calm passages \
                 shift your palette toward muted and soft; heavy passages push it to saturated and vivid. \
                 Because it is relative, a quiet solo piano uses the full range too — raise Contrast and \
                 the Attack weight to make it more sensitive to individual key strikes. Memory resets on track change.");
        }

        // Deferred actions (avoid borrowing self.config twice above)
        if enable_chroma {
            self.config.audio.chroma_enabled = true;
            self.audio_dirty = true;
        }
        if reset_memory {
            self.fx_reset = true;
            self.audio_dirty = true; // forces the app to pick the request up this frame
        }
    }
}
