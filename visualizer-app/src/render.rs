//! Offscreen compositing (GL) + export jobs.
//!
//! PARITY: preview and export both go through `Host::draw_scene` -> `Offscreen::render`, with the
//! same logical canvas (`design_scale`) and the same egui painting code. Only the pixel size differs.
//! The GL work must happen on the UI thread (it owns the context), so a running job renders a few
//! frames per UI frame (time-boxed) and hands pixels to a writer thread that feeds ffmpeg's stdin.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use eframe::egui;
use glow::HasContext;
use visualizer_core::audio::offline::{self, analysis_rate_for_fps, AnalysisTrack};
use visualizer_core::config::AppConfig;
use visualizer_core::decode::{self, TrackMeta};
use visualizer_core::export::{self as ex, RenderSettings};
use visualizer_core::project::Layer;

use crate::compose::{self, Compositor, Mode};

use crate::fxtrack;
use crate::reactive::FxState;

// ═════════════════════════ Offscreen renderer ═════════════════════════

#[derive(Clone)]
pub struct Params {
    pub px: [u32; 2],
    /// Logical canvas size in egui points (same for preview and export).
    pub logical: egui::Vec2,
    pub time: f32,
}
impl Params {
    pub fn for_settings(s: &RenderSettings, px: [u32; 2], time: f32) -> Self {
        let sc = ex::design_scale(s.width, s.height);
        Self { px, logical: egui::vec2(s.width as f32 / sc, s.height as f32 / sc), time }
    }
}

pub struct Offscreen {
    gl: Arc<glow::Context>,
    painter: egui_glow::Painter,
    pub ctx: egui::Context,
    fbo: Option<glow::Framebuffer>,
    tex: Option<glow::Texture>,
    pbos: [Option<glow::Buffer>; 2],
    pbo_pending: [bool; 2],
    pbo_next: usize,
    size: [u32; 2],
    cur_ppp: f32,
    font_key: Option<String>,
    cover_key: Option<u64>,
    cover_tex: Option<egui::TextureHandle>,
}

impl Offscreen {
    /// NOTE: `Painter::new` signature differs slightly between egui_glow versions; check 0.27.
    pub fn new(gl: Arc<glow::Context>) -> Result<Self, String> {
        let painter = egui_glow::Painter::new(gl.clone(), "", None).map_err(|e| e.to_string())?;
        Ok(Self {
            gl, painter, ctx: egui::Context::default(), fbo: None, tex: None, pbos: [None, None],
            pbo_pending: [false; 2], pbo_next: 0, size: [0, 0], cur_ppp: 0.0,
            font_key: None, cover_key: None, cover_tex: None,
        })
    }

    fn empty_frame(&self, logical: egui::Vec2) {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, logical));
        let _ = self.ctx.run(raw, |_| {});
    }

    /// Copy the main context's font definitions (custom track-info font) into this context.
    pub fn sync_fonts(
        &mut self,
        key: &str,
        defs: impl FnOnce() -> egui::FontDefinitions,
        extra: &[(String, Arc<Vec<u8>>)],
        logical: egui::Vec2,
    ) {
        let full_key = format!("{key}|{}", extra.iter().map(|(n, b)| format!("{n}:{}", b.len())).collect::<Vec<_>>().join(","));
        if self.font_key.as_deref() == Some(full_key.as_str()) { return; }
        let mut d = defs();
        for (name, bytes) in extra {
            d.font_data.insert(name.clone(), egui::FontData::from_owned(bytes.to_vec()));
            let mut fam = vec![name.clone()];
            if let Some(base) = d.families.get(&egui::FontFamily::Proportional) { fam.extend(base.clone()); } // glyph fallback
            d.families.insert(egui::FontFamily::Name(name.clone().into()), fam);
        }
        self.ctx.set_fonts(d);
        self.font_key = Some(full_key);
        self.empty_frame(logical); // fonts become active at the next frame
    }

    pub fn sync_cover(&mut self, key: u64, img: Option<&egui::ColorImage>) {
        if self.cover_key == Some(key) { return; }
        self.cover_tex = img.map(|i| self.ctx.load_texture("cover_art", i.clone(), egui::TextureOptions::LINEAR));
        self.cover_key = Some(key);
    }

    unsafe fn ensure_target(&mut self, w: u32, h: u32) -> Result<(), String> {
        if self.size == [w, h] && self.fbo.is_some() { return Ok(()); }
        let gl = &self.gl;
        if let Some(f) = self.fbo.take() { gl.delete_framebuffer(f); }
        if let Some(t) = self.tex.take() { gl.delete_texture(t); }
        for b in self.pbos.iter_mut() { if let Some(x) = b.take() { gl.delete_buffer(x); } }
        self.pbo_pending = [false; 2];

        let tex = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, w as i32, h as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, None);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        let fbo = gl.create_framebuffer()?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
        let ok = gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
        for i in 0..2 {
            let b = gl.create_buffer()?;
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(b));
            gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, (w * h * 4) as i32, glow::STREAM_READ);
            self.pbos[i] = Some(b);
        }
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        self.fbo = Some(fbo);
        self.tex = Some(tex);
        self.size = [w, h];
        if ok { Ok(()) } else { Err(format!("Cannot create a {w}x{h} render target (GPU limit?)")) }
    }

    /// Paint one frame into the FBO. `draw` gets (ctx, canvas rect, painter, cover texture).
    pub fn render(
        &mut self,
        p: &Params,
        draw: impl FnOnce(&egui::Context, egui::Rect, &egui::Painter, Option<egui::TextureHandle>),
    ) -> Result<(), String> {
        let ppp = p.px[0] as f32 / p.logical.x;
        if (self.cur_ppp - ppp).abs() > 1e-5 {
            self.ctx.set_pixels_per_point(ppp);
            self.empty_frame(p.logical); // takes effect at the next frame start
            self.cur_ppp = ppp;
        }
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, p.logical);
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(rect);
        raw.time = Some(p.time as f64);
        let cover = self.cover_tex.clone();
        let full = self.ctx.run(raw, |c| {
            let painter = c.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("scene")));
            draw(c, rect, &painter, cover);
        });
        let prims = self.ctx.tessellate(full.shapes, full.pixels_per_point);
        unsafe {
            self.ensure_target(p.px[0], p.px[1])?;
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.fbo);
            gl.viewport(0, 0, p.px[0] as i32, p.px[1] as i32);
            gl.disable(glow::SCISSOR_TEST);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.painter.paint_and_update_textures(p.px, full.pixels_per_point, &prims, &full.textures_delta);
        unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, None); } // hand the window back to eframe
        Ok(())
    }

    /// Blocking readback (preview). RGBA8, PREMULTIPLIED alpha, bottom-up rows.
    pub fn read_sync(&mut self) -> Vec<u8> {
        let [w, h] = self.size;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.fbo);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(&mut buf));
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        buf
    }

    /// Async readback for export: queues this frame, returns the PREVIOUS frame's pixels.
    pub fn push_frame(&mut self) -> Option<Vec<u8>> {
        let i = self.pbo_next;
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.fbo);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, self.pbos[i]);
            gl.read_pixels(0, 0, self.size[0] as i32, self.size[1] as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::BufferOffset(0));
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        self.pbo_pending[i] = true;
        self.pbo_next ^= 1;
        let prev = i ^ 1;
        if self.pbo_pending[prev] { self.pbo_pending[prev] = false; Some(self.map_pbo(prev)) } else { None }
    }

    /// Last queued frame (call once after the final `push_frame`).
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        for i in 0..2 { if self.pbo_pending[i] { self.pbo_pending[i] = false; return Some(self.map_pbo(i)); } }
        None
    }
    pub fn reset_readback(&mut self) { self.pbo_pending = [false; 2]; self.pbo_next = 0; }

    fn map_pbo(&self, i: usize) -> Vec<u8> {
        let n = (self.size[0] * self.size[1] * 4) as usize;
        unsafe {
            let gl = &self.gl;
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, self.pbos[i]);
            let ptr = gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, n as i32, glow::MAP_READ_BIT);
            let out = if ptr.is_null() { vec![0u8; n] } else { std::slice::from_raw_parts(ptr, n).to_vec() };
            gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            out
        }
    }
}

/// egui's output is premultiplied; ffmpeg's rgba input expects straight alpha.
pub fn unpremultiply(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a == 255 { continue; }
        if a == 0 { px[0] = 0; px[1] = 0; px[2] = 0; continue; }
        for c in 0..3 { px[c] = ((px[c] as u32 * 255 + a / 2) / a).min(255) as u8; }
    }
}

/// Bottom-up premultiplied RGBA -> top-down egui image.
pub fn to_color_image(bottom_up: &[u8], w: usize, h: usize) -> egui::ColorImage {
    let mut v = Vec::with_capacity(bottom_up.len());
    for row in (0..h).rev() { v.extend_from_slice(&bottom_up[row * w * 4..(row + 1) * w * 4]); }
    egui::ColorImage::from_rgba_premultiplied([w, h], &v)
}

// ═════════════════════════ Host (implemented by VisualizerApp) ═════════════════════════

pub struct FrameInput<'a> {
    pub main_ctx: &'a egui::Context,
    /// `None` = draw with the app's own config (preview); `Some` = swap in a job's snapshot.
    pub cfg: Option<&'a mut AppConfig>,
    /// `None` = keep the app's live bars/fx (live loopback preview).
    pub frame: Option<&'a visualizer_core::audio::BarFrame>,
    pub fx: Option<&'a FxState>,
    pub layers: &'a [Layer],
    pub comp: &'a mut Compositor,
    pub t: f64,
    pub track_len: f64,
    pub project_dir: &'a std::path::Path,
    /// A job's own cover-derived colours (swapped in for the call); `None` = the app's current ones.
    pub palettes: Option<&'a mut crate::ActivePalettes>,
    pub meta: Option<&'a TrackMeta>,
    pub cover_key: u64,
    pub cover: Option<&'a egui::ColorImage>,
    pub params: &'a Params,
    /// Opaque fill colour; `None` = transparent canvas.
    pub bg: Option<[u8; 3]>,
}

pub trait Host {
    fn draw_scene(&mut self, off: &mut Offscreen, inp: FrameInput) -> Result<(), String>;
}

// ═════════════════════════ Jobs ═════════════════════════

#[derive(Clone, Debug, PartialEq)]
pub enum JobState { Queued, Preparing, Rendering, Finishing, Done, Failed(String), Cancelled }

struct Prepared { tr: AnalysisTrack, meta: TrackMeta, cover: Option<Arc<egui::ColorImage>>, cover_small: Option<image::DynamicImage>, cover_key: u64, duration_s: f32 }
enum PrepMsg { Progress(f32), Done(Result<Prepared, String>) }

struct Run {
    prep: Prepared,
    palettes: crate::ActivePalettes,
    child: Option<Child>,
    tx: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<std::io::Result<()>>>,
    stderr: Arc<Mutex<Vec<String>>>,
    pending: Option<Vec<u8>>,
    frame: u64,
    sent: u64,
    total: u64,
    start_s: f64,
    fx: FxState,
    next_step: usize,
    started: Instant,
    part: PathBuf,
    finisher: Option<JoinHandle<Result<(), String>>>,
}

pub struct Job {
    pub id: u64,
    pub title: String,
    pub audio: PathBuf,
    pub cfg: Box<AppConfig>,
    pub settings: RenderSettings,
    pub ffmpeg: PathBuf,
    pub layers: Vec<Layer>,
    pub project_dir: PathBuf,
    comp: Compositor,
    pub out_path: PathBuf,
    pub state: JobState,
    pub progress: f32,
    pub eta_s: Option<f32>,
    pub elapsed_s: f32,
    pub attempts: u32,
    pub log: Vec<String>,
    cancel: Arc<AtomicBool>,
    prep_rx: Option<Receiver<PrepMsg>>,
    run: Option<Run>,
}

pub fn log_file_line(line: &str) {
    let dir = dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("RustVisualizer").join("logs");
    let _ = std::fs::create_dir_all(&dir);
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("render.log")) {
        let _ = writeln!(f, "[{secs}] {line}");
    }
}

fn hash_path(p: &std::path::Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.hash(&mut h);
    h.finish()
}

impl Job {
    pub fn is_active(&self) -> bool { matches!(self.state, JobState::Preparing | JobState::Rendering | JobState::Finishing) }
    pub fn is_finished(&self) -> bool { matches!(self.state, JobState::Done | JobState::Failed(_) | JobState::Cancelled) }

    fn note(&mut self, s: impl Into<String>) {
        let s = s.into();
        log_file_line(&format!("job {} '{}': {}", self.id, self.title, s));
        self.log.push(s);
        if self.log.len() > 200 { self.log.remove(0); }
    }

    fn begin_prepare(&mut self, ctx: &egui::Context) {
        self.state = JobState::Preparing;
        self.progress = 0.0;
        self.cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        self.prep_rx = Some(rx);
        let (path, cfg, cancel, rate, ctx2) = (self.audio.clone(), self.cfg.audio.clone(), self.cancel.clone(),
                                                analysis_rate_for_fps(self.settings.fps), ctx.clone());
        self.note("preparing (decode + analysis)");
        std::thread::spawn(move || {
            let res = (|| -> Result<Prepared, String> {
                let d = decode::decode_file(&path, &cancel).map_err(|e| e.to_string())?;
                let txp = tx.clone();
                let tr = offline::analyze(&d.samples, d.channels, d.sample_rate, &cfg, rate, &cancel, |p| { let _ = txp.send(PrepMsg::Progress(p)); })
                    .ok_or_else(|| "Cancelled".to_string())?;
                let decoded = d.meta.cover.as_ref().and_then(|b| image::load_from_memory(b).ok());
                let cover = decoded.as_ref().map(|img| {
                    let t = if img.width().max(img.height()) > 512 { img.thumbnail(512, 512) } else { img.clone() };
                    let rgba = t.to_rgba8();
                    Arc::new(egui::ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw()))
                });
                let cover_small = decoded.as_ref().map(|i| i.thumbnail(128, 128)); // same input the live palette code uses
                let duration_s = d.duration_s();
                Ok(Prepared { tr, meta: d.meta, cover, cover_small, cover_key: hash_path(&path), duration_s })
            })();
            let _ = tx.send(PrepMsg::Done(res));
            ctx2.request_repaint();
        });
    }

    fn start_encode(&mut self, prep: Prepared) {
        let (start, end) = self.settings.range.unwrap_or((0.0, prep.duration_s as f64));
        let dur = (end - start).max(0.05);
        let total = ((dur * self.settings.fps as f64).round() as u64).max(1);
        let ext = self.out_path.extension().and_then(|e| e.to_str()).unwrap_or("mp4").to_string();
        let part = self.out_path.with_extension(format!("{ext}.part"));
        let extras = compose::extra_audio(&self.layers, &self.project_dir, start, end);
        let args = ex::build_args(&self.settings, &self.audio, &part, start, dur, &extras);
        self.note(format!("ffmpeg {}", args.join(" ")));

        let mut cmd = Command::new(&self.ffmpeg);
        cmd.args(&args).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        ex::spawn_hidden(&mut cmd);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => { self.fail(format!("Cannot start ffmpeg: {e}")); return; }
        };
        let stdin = child.stdin.take().unwrap();
        let stderr_lines = Arc::new(Mutex::new(Vec::<String>::new()));
        if let Some(se) = child.stderr.take() {
            let sl = stderr_lines.clone();
            std::thread::spawn(move || {
                for l in BufReader::new(se).lines().flatten() {
                    let mut g = sl.lock().unwrap();
                    g.push(l);
                    if g.len() > 80 { g.remove(0); }
                }
            });
        }
        let (tx, rx) = sync_channel::<Vec<u8>>(3);
        let writer = std::thread::spawn(move || -> std::io::Result<()> {
            let mut stdin = stdin;
            for buf in rx { stdin.write_all(&buf)?; }
            stdin.flush()
        });

        // Deterministic FX: replay from t = 0 up to the range start.
        let rate = prep.tr.rate_hz;
        let start_step = ((start * rate as f64).round() as usize).min(prep.tr.step_count().saturating_sub(1));
        let mut fx = FxState::new();
        for i in 0..start_step { fxtrack::step(&mut fx, &prep.tr, &self.cfg, i); }

        let palettes = crate::compute_palettes_sync(&self.cfg, prep.cover_small.as_ref());
        self.run = Some(Run { prep, palettes, child: Some(child), tx: Some(tx), writer: Some(writer), stderr: stderr_lines, pending: None,
            frame: 0, sent: 0, total, start_s: start, fx, next_step: start_step, started: Instant::now(), part, finisher: None });
        self.state = JobState::Rendering;
        self.progress = 0.0;
    }

    fn tail(stderr: &Arc<Mutex<Vec<String>>>) -> String {
        let g = stderr.lock().unwrap();
        let n = g.len().saturating_sub(6);
        g[n..].join(" | ")
    }

    /// Stop everything that is running for this job and delete the partial file.
    fn teardown(&mut self) {
        if let Some(mut run) = self.run.take() {
            run.tx.take();
            if let Some(mut c) = run.child.take() { let _ = c.kill(); let part = run.part.clone(); let w = run.writer.take();
                std::thread::spawn(move || { if let Some(w) = w { let _ = w.join(); } let _ = c.wait(); let _ = std::fs::remove_file(part); }); }
        }
    }
    fn fail(&mut self, msg: String) {
        let tail = self.run.as_ref().map(|r| Self::tail(&r.stderr)).unwrap_or_default();
        let full = if tail.is_empty() { msg } else { format!("{msg} (ffmpeg: {tail})") };
        self.note(format!("FAILED: {full}"));
        self.teardown();
        self.state = JobState::Failed(full);
    }
    pub fn cancel(&mut self) {
        if self.is_finished() { return; }
        self.cancel.store(true, Ordering::Relaxed);
        self.teardown();
        self.note("cancelled");
        self.state = JobState::Cancelled;
    }
    pub fn retry(&mut self) {
        if matches!(self.state, JobState::Failed(_) | JobState::Cancelled) {
            self.attempts += 1;
            self.state = JobState::Queued;
            self.progress = 0.0;
            self.eta_s = None;
            self.note(format!("retry #{}", self.attempts));
        }
    }

    fn step(&mut self, host: &mut dyn Host, off: &mut Offscreen, ctx: &egui::Context) {
        match self.state {
            JobState::Preparing => {
                let mut done: Option<Result<Prepared, String>> = None;
                if let Some(rx) = &self.prep_rx {
                    while let Ok(m) = rx.try_recv() {
                        match m { PrepMsg::Progress(p) => self.progress = p, PrepMsg::Done(r) => done = Some(r) }
                    }
                }
                match done {
                    Some(Ok(p)) => { self.prep_rx = None; self.start_encode(p); }
                    Some(Err(e)) => { self.prep_rx = None;
                        if self.cancel.load(Ordering::Relaxed) { self.state = JobState::Cancelled; } else { self.fail(e); } }
                    None => {}
                }
            }
            JobState::Rendering => self.render_slice(host, off, ctx),
            JobState::Finishing => {
                let fin = match self.run.as_mut() {
                    Some(r) if r.finisher.as_ref().map(|h| h.is_finished()).unwrap_or(false) => r.finisher.take(),
                    _ => None,
                };
                if let Some(h) = fin {
                    match h.join().unwrap_or_else(|_| Err("finisher panicked".into())) {
                        Ok(()) => { self.progress = 1.0; self.eta_s = Some(0.0); self.note(format!("done -> {}", self.out_path.display())); self.run = None; self.state = JobState::Done; }
                        Err(e) => self.fail(e),
                    }
                }
            }
            _ => {}
        }
    }

    fn render_slice(&mut self, host: &mut dyn Host, off: &mut Offscreen, ctx: &egui::Context) {
        let deadline = Instant::now() + Duration::from_millis(12);
        let settings = self.settings.clone();
        let (w, h) = (settings.width, settings.height);
        let bg = if settings.transparent { None } else { Some(settings.background) };

        loop {
            let Some(run) = self.run.as_mut() else { return };
            if self.cancel.load(Ordering::Relaxed) { return; }
            if run.writer.as_ref().map(|w| w.is_finished()).unwrap_or(true) {
                self.fail("ffmpeg stopped unexpectedly".into());
                return;
            }
            // 1. deliver a buffer that did not fit last time
            if let Some(buf) = run.pending.take() {
                match run.tx.as_ref().unwrap().try_send(buf) {
                    Ok(()) => run.sent += 1,
                    Err(TrySendError::Full(b)) => { run.pending = Some(b); return; } // encoder is the bottleneck; retry next UI frame
                    Err(TrySendError::Disconnected(_)) => { self.fail("ffmpeg closed its input".into()); return; }
                }
            }
            // 2. all frames rendered -> flush the pipeline and finish
            if run.frame >= run.total {
                if let Some(mut px) = off.flush() {
                    if settings.transparent { unpremultiply(&mut px); }
                    run.pending = Some(px);
                    continue;
                }
                let (child, writer, stderr, part, out) = (run.child.take(), run.writer.take(), run.stderr.clone(), run.part.clone(), self.out_path.clone());
                run.tx.take(); // EOF for the writer thread
                let overwrite = settings.collision == ex::Collision::Overwrite;
                run.finisher = Some(std::thread::spawn(move || {
                    let w = writer.map(|w| w.join());
                    let st = child.map(|mut c| c.wait());
                    std::thread::sleep(Duration::from_millis(80));
                    let ok = matches!(w, Some(Ok(Ok(())))) && matches!(st, Some(Ok(s)) if s.success());
                    if ok {
                        if overwrite { let _ = std::fs::remove_file(&out); }
                        std::fs::rename(&part, &out).map_err(|e| format!("Cannot move the finished file into place: {e}"))
                    } else {
                        let _ = std::fs::remove_file(&part);
                        Err(format!("ffmpeg failed: {}", Self::tail(&stderr)))
                    }
                }));
                self.state = JobState::Finishing;
                self.progress = 0.99;
                off.reset_readback();
                return;
            }
            if Instant::now() > deadline { ctx.request_repaint(); return; }

            // 3. render frame k
            let k = run.frame;
            let t = run.start_s + k as f64 / settings.fps as f64;
            let rate = run.prep.tr.rate_hz as f64;
            let target = ((t * rate).round() as usize).min(run.prep.tr.step_count() - 1);
            while run.next_step <= target {
                fxtrack::step(&mut run.fx, &run.prep.tr, &self.cfg, run.next_step);
                run.next_step += 1;
            }
            let track_len = run.prep.duration_s as f64;
            // Background videos must have frame k ready before we draw it (retry next UI frame otherwise).
            if !self.comp.prepare(&self.layers, t, track_len, &self.project_dir) { return; }
            let frame = run.prep.tr.frames[target].clone();
            let params = Params::for_settings(&settings, [w, h], t as f32);
            let res = host.draw_scene(off, FrameInput {
                main_ctx: ctx, cfg: Some(&mut *self.cfg), frame: Some(&frame), fx: Some(&run.fx),
                layers: &self.layers, comp: &mut self.comp, t, track_len, project_dir: &self.project_dir,
                palettes: Some(&mut run.palettes), meta: Some(&run.prep.meta), cover_key: run.prep.cover_key, cover: run.prep.cover.as_deref(),
                params: &params, bg,
            });
            if let Err(e) = res { self.fail(e); return; }
            run.frame += 1;
            if let Some(mut px) = off.push_frame() {
                if settings.transparent { unpremultiply(&mut px); }
                run.pending = Some(px);
            }
            let el = run.started.elapsed().as_secs_f32();
            self.progress = (run.frame as f32 / run.total as f32).min(0.99);
            self.elapsed_s = el;
            self.eta_s = (run.frame > 5).then(|| el / run.frame as f32 * (run.total - run.frame) as f32);
        }
    }
}

// ═════════════════════════ Queue ═════════════════════════

pub struct RenderQueue {
    pub jobs: Vec<Job>,
    next_id: u64,
}

impl RenderQueue {
    pub fn new() -> Self { Self { jobs: vec![], next_id: 1 } }

    pub fn taken_paths(&self) -> HashSet<PathBuf> {
        self.jobs.iter().filter(|j| !matches!(j.state, JobState::Failed(_) | JobState::Cancelled)).map(|j| j.out_path.clone()).collect()
    }

    pub fn add(&mut self, title: String, audio: PathBuf, cfg: AppConfig, settings: RenderSettings, ffmpeg: PathBuf, out_path: PathBuf,
               layers: Vec<Layer>, project_dir: PathBuf) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let comp = Compositor::new(Mode::Export, Some(ffmpeg.clone()), settings.fps, 3840, settings.hw_decode);
        let mut j = Job { id, title, audio, cfg: Box::new(cfg), settings, ffmpeg, layers, project_dir, comp, out_path, state: JobState::Queued, progress: 0.0,
            eta_s: None, elapsed_s: 0.0, attempts: 0, log: vec![], cancel: Arc::new(AtomicBool::new(false)), prep_rx: None, run: None };
        j.note("queued");
        self.jobs.push(j);
        id
    }
    pub fn remove(&mut self, id: u64) {
        if let Some(j) = self.jobs.iter_mut().find(|j| j.id == id) { j.cancel(); }
        self.jobs.retain(|j| j.id != id);
    }
    pub fn move_job(&mut self, id: u64, d: i32) {
        if let Some(i) = self.jobs.iter().position(|j| j.id == id) {
            if self.jobs[i].is_active() { return; }
            let k = (i as i32 + d).clamp(0, self.jobs.len() as i32 - 1) as usize;
            if !self.jobs[k].is_active() { self.jobs.swap(i, k); }
        }
    }
    pub fn clear_finished(&mut self) { self.jobs.retain(|j| !matches!(j.state, JobState::Done)); }
    pub fn busy(&self) -> bool { self.jobs.iter().any(|j| j.is_active()) }
    pub fn shutdown(&mut self) { for j in self.jobs.iter_mut() { j.cancel(); } }

    /// Call once per UI frame. Runs one job at a time, in list order.
    pub fn pump(&mut self, host: &mut dyn Host, off: &mut Offscreen, ctx: &egui::Context) {
        if !self.busy() {
            if let Some(j) = self.jobs.iter_mut().find(|j| j.state == JobState::Queued) { j.begin_prepare(ctx); }
        }
        for j in self.jobs.iter_mut().filter(|j| j.is_active()) { j.step(host, off, ctx); }
        if self.busy() { ctx.request_repaint(); }
    }
}
