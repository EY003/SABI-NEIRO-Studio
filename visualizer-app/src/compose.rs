//! Layer compositing inside the offscreen egui context: image / video / text / visualizer layers.
//! One `Compositor` per consumer (preview, or one per export job): they own their video decoders.
//! Preview and export differ only in `Mode` (random-access vs strictly sequential video) and
//! the decode resolution; the drawing code is shared, so layering, fit modes, fades and text
//! placement are identical.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use visualizer_core::export::{ExtraAudio, Issue, Severity};
use visualizer_core::project::{FitMode, Layer, LayerKind, MediaRef, TextAlign, TextLayer};

use crate::video::{self, VideoInfo, VideoStream};

// ── pure helpers (unit tested) ───────────────────────────────

/// Where an image of size `src` lands on `canvas`. `offset` is a fraction of the canvas size.
/// The caller clips to the canvas, so Cover simply returns a rect larger than the canvas.
pub fn fit_rect(src: [f32; 2], canvas: egui::Rect, mode: FitMode, offset: [f32; 2], scale: f32) -> egui::Rect {
    let c = canvas.size();
    let (sw, sh) = (src[0].max(1.0), src[1].max(1.0));
    let size = match mode {
        FitMode::Stretch => c,
        FitMode::Contain => { let s = (c.x / sw).min(c.y / sh); egui::vec2(sw * s, sh * s) }
        FitMode::Cover => { let s = (c.x / sw).max(c.y / sh); egui::vec2(sw * s, sh * s) }
    };
    egui::Rect::from_center_size(canvas.center() + egui::vec2(offset[0] * c.x, offset[1] * c.y), size * scale.max(0.01))
}

pub fn font_family_name(path: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    format!("tl_{:x}", h.finish())
}

fn alpha_u8(a: f32) -> u8 { (a.clamp(0.0, 1.0) * 255.0).round() as u8 }

// ── video layer state ────────────────────────────────────────

#[derive(PartialEq)]
struct VidKey { path: PathBuf, trim_bits: u64, looping: bool }

struct VidState {
    key: Option<VidKey>,
    info: Option<VideoInfo>,
    stream: Option<VideoStream>,
    cur_k: i64,
    pending: Option<Vec<u8>>,
    dims: (usize, usize),
    tex: Option<egui::TextureHandle>,
    last_open: Instant,
    failed: bool,
}
impl VidState {
    fn new() -> Self {
        Self { key: None, info: None, stream: None, cur_k: -1, pending: None, dims: (2, 2), tex: None,
               last_open: Instant::now() - Duration::from_secs(10), failed: false }
    }
    fn reset(&mut self) { *self = Self::new(); }
}

struct ImgEntry { tex: egui::TextureHandle, size: [f32; 2] }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode { Preview, Export }

pub struct Compositor {
    mode: Mode,
    pub ffmpeg: Option<PathBuf>,
    fps: f32,
    max_w: usize,
    hwdec: bool,
    images: HashMap<PathBuf, Option<ImgEntry>>,
    videos: HashMap<u64, VidState>,
    fonts: HashMap<String, Option<Arc<Vec<u8>>>>,
    pub errors: Vec<String>,
}

impl Compositor {
    pub fn new(mode: Mode, ffmpeg: Option<PathBuf>, fps: f32, max_w: usize, hwdec: bool) -> Self {
        Self { mode, ffmpeg, fps, max_w, hwdec, images: HashMap::new(), videos: HashMap::new(), fonts: HashMap::new(), errors: vec![] }
    }
    pub fn set_hwdec(&mut self, on: bool) { if self.hwdec != on { self.hwdec = on; self.videos.clear(); } }
    pub fn set_fps(&mut self, fps: f32) {
        if (self.fps - fps).abs() > 1e-4 { self.fps = fps; self.videos.clear(); }
    }

    /// Fonts the text layers need, as (family name, bytes), for `Offscreen::sync_fonts`.
    pub fn font_extras(&mut self, layers: &[Layer]) -> Vec<(String, Arc<Vec<u8>>)> {
        let mut out = vec![];
        for l in layers {
            if let LayerKind::Text(t) = &l.kind {
                if let Some(p) = t.font_path.as_ref().filter(|p| !p.is_empty()) {
                    let bytes = self.fonts.entry(p.clone()).or_insert_with(|| std::fs::read(p).ok().map(Arc::new)).clone();
                    if let Some(b) = bytes {
                        let name = font_family_name(p);
                        if !out.iter().any(|(n, _): &(String, Arc<Vec<u8>>)| *n == name) { out.push((name, b)); }
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Advance video decoders to time `t`. Export mode returns `false` while a frame is not
    /// ready yet (the caller retries next UI frame); preview never blocks.
    pub fn prepare(&mut self, layers: &[Layer], t: f64, track_len: f64, dir: &Path) -> bool {
        let mut ready = true;
        let mut alive: Vec<u64> = vec![];
        for l in layers {
            if let LayerKind::Video { media, looping, trim_start_s, .. } = &l.kind {
                if l.alpha_at(t, track_len) <= 0.0 { continue; } // hidden / outside span: decoder is dropped below
                alive.push(l.id);
                if !self.ensure_video(l, media, *trim_start_s, *looping, t, dir) { ready = false; }
            }
        }
        self.videos.retain(|id, _| alive.contains(id));
        ready
    }

    fn ensure_video(&mut self, l: &Layer, media: &MediaRef, trim: f64, looping: bool, t: f64, dir: &Path) -> bool {
        let Some(path) = media.resolve(dir) else { return true };
        let Some(ff) = self.ffmpeg.clone() else { return true };
        let fps = self.fps;
        let k = ((t - l.start_s).max(0.0) * fps as f64).floor() as i64;
        let mode = self.mode;
        let hwdec = self.hwdec;
        let st = self.videos.entry(l.id).or_insert_with(VidState::new);
        if st.failed { return true; }
        let key = VidKey { path: path.clone(), trim_bits: trim.to_bits(), looping };
        if st.key.as_ref() != Some(&key) { st.reset(); st.key = Some(key); }
        if st.info.is_none() {
            match video::probe(&ff, &path) {
                Ok(i) => st.info = Some(i),
                Err(e) => { st.failed = true; self.errors.push(e); return true; }
            }
        }
        let info = st.info.unwrap();
        let (ow, oh) = video::scaled_size(info.width, info.height, self.max_w);
        let clip = info.duration_s.map(|d| (d - trim).max(0.1));
        let media_time = |k: i64| -> f64 {
            let local = k as f64 / fps as f64;
            trim + match (looping, clip) { (true, Some(c)) => local % c, _ => local }
        };

        let need_open = match &st.stream {
            None => true,
            Some(_) => mode == Mode::Preview && (k < st.cur_k || k > st.cur_k + 90),
        };
        if need_open {
            if mode == Mode::Preview && st.stream.is_some() && st.last_open.elapsed() < Duration::from_millis(150) { return true; } // throttle scrubbing
            match VideoStream::open(&ff, &path, media_time(k.max(0)), fps, ow, oh, hwdec) {
                Ok(s) => { st.stream = Some(s); st.cur_k = k - 1; st.last_open = Instant::now(); }
                Err(e) => { st.failed = true; self.errors.push(e); return true; }
            }
        }
        let mut spins = 0;
        while st.cur_k < k && spins < 600 {
            spins += 1;
            let Some(stream) = st.stream.as_mut() else { break };
            match stream.try_next() {
                Some(buf) => { st.pending = Some(buf); st.cur_k += 1; st.dims = (ow, oh); }
                None if stream.is_finished() => {
                    if looping {
                        match VideoStream::open(&ff, &path, trim, fps, ow, oh, hwdec) { Ok(s) => st.stream = Some(s), Err(_) => break }
                    } else { break } // hold the last frame
                }
                None => return mode == Mode::Preview, // decoder still working
            }
        }
        true
    }

    fn image_tex(&mut self, c: &egui::Context, media: &MediaRef, dir: &Path) -> Option<(egui::TextureId, [f32; 2])> {
        let path = media.resolve(dir)?;
        if !self.images.contains_key(&path) {
            let entry = match image::open(&path) {
                Ok(img) => {
                    let img = if img.width().max(img.height()) > 4096 { img.thumbnail(4096, 4096) } else { img };
                    let rgba = img.to_rgba8();
                    let size = [rgba.width() as usize, rgba.height() as usize];
                    let tex = c.load_texture("layer_img", egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()), egui::TextureOptions::LINEAR);
                    Some(ImgEntry { tex, size: [size[0] as f32, size[1] as f32] })
                }
                Err(e) => { self.errors.push(format!("{}: {e}", path.display())); None }
            };
            self.images.insert(path.clone(), entry);
        }
        self.images.get(&path).and_then(|e| e.as_ref()).map(|e| (e.tex.id(), e.size))
    }

    /// Draw every layer, bottom to top. `visualizer(painter, alpha)` draws bars + track info; the app
    /// applies `alpha` by scaling its own opacity settings (egui 0.27 has no per-painter opacity).
    pub fn draw_layers(
        &mut self,
        c: &egui::Context,
        painter: &egui::Painter,
        canvas: egui::Rect,
        layers: &[Layer],
        t: f64,
        track_len: f64,
        dir: &Path,
        visualizer: &mut dyn FnMut(&egui::Painter, f32),
    ) {
        let clipped = painter.with_clip_rect(canvas);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        for l in layers {
            let a = l.alpha_at(t, track_len);
            if a <= 0.0 { continue; }
            let tint = egui::Color32::from_white_alpha(alpha_u8(a));
            match &l.kind {
                LayerKind::Visualizer => visualizer(painter, a),
                LayerKind::Image { media, fit, offset, scale } => {
                    if let Some((id, size)) = self.image_tex(c, media, dir) {
                        clipped.image(id, fit_rect(size, canvas, *fit, *offset, *scale), uv, tint);
                    }
                }
                LayerKind::Video { fit, offset, scale, .. } => {
                    if let Some(st) = self.videos.get_mut(&l.id) {
                        if let Some(buf) = st.pending.take() {
                            let img = egui::ColorImage::from_rgba_unmultiplied([st.dims.0, st.dims.1], &buf);
                            match &mut st.tex {
                                Some(tx) => tx.set(img, egui::TextureOptions::LINEAR),
                                None => st.tex = Some(c.load_texture(format!("layer_vid_{}", l.id), img, egui::TextureOptions::LINEAR)),
                            }
                        }
                        if let Some(tx) = &st.tex {
                            clipped.image(tx.id(), fit_rect([st.dims.0 as f32, st.dims.1 as f32], canvas, *fit, *offset, *scale), uv, tint);
                        }
                    }
                }
                LayerKind::Text(tl) => draw_text(&clipped, canvas, tl, a),
            }
        }
    }
}

fn draw_text(p: &egui::Painter, canvas: egui::Rect, tl: &TextLayer, a: f32) {
    if tl.text.is_empty() { return; }
    let family = match tl.font_path.as_ref().filter(|s| !s.is_empty()) {
        Some(path) => egui::FontFamily::Name(font_family_name(path).into()),
        None => egui::FontFamily::Proportional,
    };
    let size = (tl.size * canvas.height()).max(4.0);
    let pos = canvas.min + egui::vec2(tl.pos[0] * canvas.width(), tl.pos[1] * canvas.height());
    let align = match tl.align { TextAlign::Left => egui::Align2::LEFT_CENTER, TextAlign::Center => egui::Align2::CENTER_CENTER, TextAlign::Right => egui::Align2::RIGHT_CENTER };
    let col = egui::Color32::from_rgba_unmultiplied(tl.color[0], tl.color[1], tl.color[2], alpha_u8(a));
    let shaped = crate::textshape::shape(&tl.text);
    let text: &str = shaped.as_ref();
    let font = egui::FontId::new(size, family);
    let draw = |offset: egui::Vec2, color: egui::Color32| { p.text(pos + offset, align, text, font.clone(), color); };
    // egui has no outline/shadow for text, so both are built from offset copies of the same text.
    // Offsets are in canvas points (short side = 1080), identical in preview and export.

    if tl.shadow && tl.shadow_opacity > 0.0 {
        let base = (tl.shadow_opacity.clamp(0.0, 1.0) * a).clamp(0.0, 1.0);
        let off = egui::vec2(tl.shadow_offset[0], tl.shadow_offset[1]);
        let rgb = tl.shadow_color;
        if tl.shadow_blur > 0.5 {
            // soft shadow: faint copies spread over two rings; together they add up to `base` at the centre
            let mut spots = vec![egui::Vec2::ZERO];
            for (ring, r) in [(8usize, tl.shadow_blur * 0.5), (12usize, tl.shadow_blur)] {
                for i in 0..ring {
                    let ang = i as f32 / ring as f32 * std::f32::consts::TAU;
                    spots.push(egui::vec2(ang.cos() * r, ang.sin() * r));
                }
            }
            let per = 1.0 - (1.0 - base).powf(1.0 / spots.len() as f32);
            let col = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], alpha_u8(per));
            for s in spots { draw(off + s, col); }
        } else {
            draw(off, egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], alpha_u8(base)));
        }
    }
    if tl.outline_width > 0.1 {
        let w = tl.outline_width;
        let oc = tl.outline_color;
        let col = egui::Color32::from_rgba_unmultiplied(oc[0], oc[1], oc[2], alpha_u8(a));
        let steps = ((8.0 + 4.0 * w).round() as usize).clamp(12, 40);
        let rings: &[f32] = if w > 3.0 { &[1.0, 0.5] } else { &[1.0] }; // thick outlines get an inner ring so there are no gaps
        for k in rings {
            for i in 0..steps {
                let ang = i as f32 / steps as f32 * std::f32::consts::TAU;
                draw(egui::vec2(ang.cos(), ang.sin()) * (w * k), col);
            }
        }
    }
    draw(egui::Vec2::ZERO, col);
}

// ── export-side helpers ──────────────────────────────────────

/// Sound of un-muted, non-looping background videos, positioned for an export of [r0, r1].
pub fn extra_audio(layers: &[Layer], dir: &Path, r0: f64, r1: f64) -> Vec<ExtraAudio> {
    let mut v = vec![];
    for l in layers.iter().filter(|l| l.visible) {
        if let LayerKind::Video { media, mute: false, looping: false, trim_start_s, .. } = &l.kind {
            let end = l.end_s.unwrap_or(f64::MAX);
            if end <= r0 || l.start_s >= r1 { continue; }
            if let Some(path) = media.resolve(dir) {
                v.push(ExtraAudio { path, seek_s: trim_start_s + (r0 - l.start_s).max(0.0), delay_s: (l.start_s - r0).max(0.0), volume: 1.0 });
            }
        }
    }
    v
}

/// Problems with layer media that should block or warn before a long render starts.
pub fn layer_issues(layers: &[Layer], dir: &Path, ffmpeg_known: bool) -> Vec<Issue> {
    let mut v = vec![];
    let err = |m: String| Issue { severity: Severity::Error, message: m };
    let warn = |m: String| Issue { severity: Severity::Warning, message: m };
    if !layers.iter().any(|l| matches!(l.kind, LayerKind::Visualizer) && l.visible) {
        v.push(warn("The visualizer layer is hidden or missing: the output will have no bars.".into()));
    }
    for l in layers {
        match &l.kind {
            LayerKind::Image { media, .. } | LayerKind::Video { media, .. } => {
                if media.resolve(dir).is_none() { v.push(err(format!("Layer '{}': file is missing ({}). Relink it in File > Missing files.", l.name, media.path.display()))); }
            }
            LayerKind::Text(t) => {
                if let Some(p) = t.font_path.as_ref().filter(|p| !p.is_empty()) {
                    if !Path::new(p).is_file() { v.push(warn(format!("Layer '{}': font file not found, the default font will be used.", l.name))); }
                }
            }
            LayerKind::Visualizer => {}
        }
        if let LayerKind::Video { looping: true, mute: false, .. } = &l.kind {
            v.push(warn(format!("Layer '{}': looped video sound is not mixed into the export (mute it or turn looping off).", l.name)));
        }
        if matches!(l.kind, LayerKind::Video { .. }) && !ffmpeg_known {
            v.push(err(format!("Layer '{}': ffmpeg is needed to decode video.", l.name)));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    fn canvas() -> egui::Rect { egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 200.0)) }
    #[test]
    fn contain_and_cover() {
        let r = fit_rect([100.0, 50.0], canvas(), FitMode::Contain, [0.0, 0.0], 1.0);
        assert_eq!((r.width(), r.height()), (200.0, 100.0));
        assert_eq!(r.center(), egui::pos2(100.0, 100.0));
        let r = fit_rect([100.0, 50.0], canvas(), FitMode::Cover, [0.0, 0.0], 1.0);
        assert_eq!(fit_rect([100.0, 50.0], canvas(), FitMode::Contain, [0.0, 0.0], 0.5).width(), 100.0);
        assert_eq!((r.width(), r.height()), (400.0, 200.0));
        let r = fit_rect([100.0, 50.0], canvas(), FitMode::Stretch, [0.1, 0.0], 1.0);
        assert_eq!(r.center(), egui::pos2(120.0, 100.0));
    }
}
