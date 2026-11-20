//! Audio output with a sample-counter master clock.
//!
//! The playhead is the number of SOURCE frames the output callback has consumed (minus the
//! device's reported output latency), so visuals follow what is audible, and seeking,
//! pausing and track switching can never drift: there is no wall-clock timer anywhere.
//! The cpal `Stream` is not `Send`, so it lives on its own thread; `Player` is `Send + Sync`.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

pub struct Track {
    pub samples: Vec<f32>, // interleaved
    pub channels: usize,
    pub sample_rate: u32,
}
impl Track {
    pub fn frames(&self) -> usize { self.samples.len() / self.channels.max(1) }
    pub fn duration_s(&self) -> f32 { self.frames() as f32 / self.sample_rate.max(1) as f32 }
}

struct Shared {
    track: ArcSwapOption<Track>,
    pos: AtomicU64, // f64 bits, in source frames
    playing: AtomicBool,
    ended: AtomicBool,
    volume: AtomicU32, // f32 bits
    latency_ns: AtomicU64,
    quit: AtomicBool,
    error: Mutex<Option<String>>,
}

pub struct Player {
    sh: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl Player {
    pub fn new() -> Self {
        let sh = Arc::new(Shared {
            track: ArcSwapOption::empty(),
            pos: AtomicU64::new(0f64.to_bits()),
            playing: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            volume: AtomicU32::new(1.0f32.to_bits()),
            latency_ns: AtomicU64::new(0),
            quit: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let sh2 = Arc::clone(&sh);
        let thread = std::thread::Builder::new().name("audio-out".into()).spawn(move || {
            if let Err(e) = run_output(&sh2) {
                *sh2.error.lock().unwrap() = Some(e);
            }
        }).ok();
        Self { sh, thread }
    }

    pub fn error(&self) -> Option<String> { self.sh.error.lock().ok().and_then(|e| e.clone()) }

    /// Replaces the track; position 0, paused.
    pub fn set_track(&self, t: Option<Arc<Track>>) {
        self.sh.playing.store(false, Relaxed);
        self.sh.track.store(t);
        self.sh.pos.store(0f64.to_bits(), Relaxed);
        self.sh.ended.store(false, Relaxed);
    }
    pub fn play(&self) {
        if self.sh.track.load().is_none() { return; }
        if self.sh.ended.swap(false, Relaxed) { self.sh.pos.store(0f64.to_bits(), Relaxed); }
        self.sh.playing.store(true, Relaxed);
    }
    pub fn pause(&self) { self.sh.playing.store(false, Relaxed); }
    pub fn toggle(&self) { if self.is_playing() { self.pause() } else { self.play() } }
    pub fn is_playing(&self) -> bool { self.sh.playing.load(Relaxed) }
    pub fn ended(&self) -> bool { self.sh.ended.load(Relaxed) }
    pub fn set_volume(&self, v: f32) { self.sh.volume.store(v.clamp(0.0, 1.5).to_bits(), Relaxed); }
    pub fn volume(&self) -> f32 { f32::from_bits(self.sh.volume.load(Relaxed)) }

    pub fn seek(&self, secs: f32) {
        if let Some(t) = self.sh.track.load_full() {
            let f = (secs.max(0.0) as f64 * t.sample_rate as f64).min(t.frames().saturating_sub(2) as f64);
            self.sh.pos.store(f.to_bits(), Relaxed);
            self.sh.ended.store(false, Relaxed);
        }
    }

    /// What the listener is hearing right now (device latency subtracted while playing).
    pub fn position_s(&self) -> f32 {
        let Some(t) = self.sh.track.load_full() else { return 0.0 };
        let p = f64::from_bits(self.sh.pos.load(Relaxed)) / t.sample_rate.max(1) as f64;
        let lat = if self.is_playing() { self.sh.latency_ns.load(Relaxed) as f64 / 1e9 } else { 0.0 };
        (p - lat).max(0.0) as f32
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.sh.quit.store(true, Relaxed);
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}

fn run_output(sh: &Arc<Shared>) -> Result<(), String> {
    let host = cpal::default_host();
    let dev = host.default_output_device().ok_or("No audio output device")?;
    let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
    match cfg.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&dev, cfg, sh),
        cpal::SampleFormat::I16 => run::<i16>(&dev, cfg, sh),
        cpal::SampleFormat::U16 => run::<u16>(&dev, cfg, sh),
        f => Err(format!("Unsupported output sample format {f:?}")),
    }
}

fn run<T: SizedSample + FromSample<f32>>(
    dev: &cpal::Device,
    cfg: cpal::SupportedStreamConfig,
    sh: &Arc<Shared>,
) -> Result<(), String> {
    let ch = cfg.channels() as usize;
    let dev_sr = cfg.sample_rate().0 as f64;
    let stream_cfg: cpal::StreamConfig = cfg.into();
    let shc = Arc::clone(sh);
    let mut scratch: Vec<f32> = Vec::new();
    let stream = dev.build_output_stream(
        &stream_cfg,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            fill(&shc, ch, dev_sr, &mut scratch, data.len() / ch.max(1));
            for (o, s) in data.iter_mut().zip(scratch.iter()) { *o = T::from_sample(*s); }
            let ts = info.timestamp();
            if let Some(d) = ts.playback.duration_since(&ts.callback) {
                shc.latency_ns.store(d.as_nanos() as u64, Relaxed);
            }
        },
        |e| eprintln!("audio output error: {e}"),
        None,
    ).map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    while !sh.quit.load(Relaxed) { std::thread::sleep(Duration::from_millis(50)); }
    drop(stream);
    Ok(())
}

/// Fill `frames` device frames. Linear-interpolating resampler (source rate -> device rate).
fn fill(sh: &Shared, ch: usize, dev_sr: f64, out: &mut Vec<f32>, frames: usize) {
    out.clear();
    out.resize(frames * ch, 0.0);
    if !sh.playing.load(Relaxed) { return; }
    let Some(track) = sh.track.load_full() else { return };
    let start_bits = sh.pos.load(Relaxed);
    let mut pos = f64::from_bits(start_bits);
    let step = track.sample_rate as f64 / dev_sr;
    let src = track.channels.max(1);
    let n = track.frames() as f64;
    let vol = f32::from_bits(sh.volume.load(Relaxed));
    let mut hit_end = false;

    for f in 0..frames {
        if pos >= n - 1.0 { hit_end = true; break; }
        let i = pos as usize;
        let fr = (pos - i as f64) as f32;
        let a = &track.samples[i * src..(i + 1) * src];
        let b = &track.samples[(i + 1) * src..(i + 2) * src];
        let s = |c: usize| a[c] + (b[c] - a[c]) * fr;
        let o = &mut out[f * ch..(f + 1) * ch];
        if ch == 1 {
            o[0] = (0..src).map(s).sum::<f32>() / src as f32 * vol;
        } else if src == 1 {
            let v = s(0) * vol;
            for x in o.iter_mut() { *x = v; }
        } else {
            o[0] = s(0) * vol;
            o[1] = s(1) * vol;
        }
        pos += step;
    }
    // If the UI seeked during this callback, the CAS fails and our advance is discarded.
    if sh.pos.compare_exchange(start_bits, pos.to_bits(), Relaxed, Relaxed).is_ok() && hit_end {
        sh.ended.store(true, Relaxed);
        sh.playing.store(false, Relaxed);
    }
}
