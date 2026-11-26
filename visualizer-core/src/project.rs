//! Project document: everything that is saved to a project file and that undo/redo covers.
//! Layer order: index 0 is the BOTTOM layer.
//! `ProjectDoc` is Clone-only on purpose (AppConfig has no PartialEq); change detection is done by
//! the caller (config fingerprint) and recorded with `UndoStack::push`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::AppConfig;
use crate::export::RenderSettings;

pub const PROJECT_VERSION: u32 = 1;

fn default_scale() -> f32 { 1.0 }

// ── Media references (relinking / portability) ───────────────

/// Equality is by path only: `rel`/`size` are bookkeeping and must not count as an edit.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MediaRef {
    pub path: PathBuf,
    /// Path relative to the project file, when the media lives under it.
    #[serde(default)] pub rel: Option<PathBuf>,
    #[serde(default)] pub size: u64,
}

impl PartialEq for MediaRef { fn eq(&self, o: &Self) -> bool { self.path == o.path } }

impl MediaRef {
    pub fn new(path: &Path, project_dir: Option<&Path>) -> Self {
        let rel = project_dir.and_then(|d| path.strip_prefix(d).ok()).map(|p| p.to_path_buf());
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Self { path: path.to_path_buf(), rel, size }
    }

    /// Absolute path, then relative-to-project, then same file name next to the project
    /// (accepted only if the size matches). `None` means "missing: ask the user to relink".
    pub fn resolve(&self, project_dir: &Path) -> Option<PathBuf> {
        if self.path.is_file() { return Some(self.path.clone()); }
        if let Some(rel) = &self.rel {
            let p = project_dir.join(rel);
            if p.is_file() { return Some(p); }
        }
        let name = self.path.file_name()?;
        let p = project_dir.join(name);
        if p.is_file() && (self.size == 0 || std::fs::metadata(&p).map(|m| m.len()).ok() == Some(self.size)) {
            return Some(p);
        }
        None
    }

    /// Offer: search `dir` for a file with the same name (used after the user picks a folder).
    pub fn relink_in(&mut self, dir: &Path) -> bool {
        if let Some(name) = self.path.file_name() {
            let p = dir.join(name);
            if p.is_file() { self.path = p; return true; }
        }
        false
    }
}

// ── Canvas / layers ──────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FitMode { #[default] Cover, Contain, Stretch }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TextAlign { Left, #[default] Center, Right }

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct TextLayer {
    pub text: String,
    pub font_path: Option<String>,
    /// Size as a fraction of canvas height, so text scales with resolution.
    pub size: f32,
    pub color: [u8; 3],
    pub align: TextAlign,
    /// Anchor position, 0..1 of the canvas.
    pub pos: [f32; 2],
    /// Outline thickness in canvas points (the short canvas side is 1080 points); 0 = none.
    pub outline_width: f32,
    pub outline_color: [u8; 3],
    pub shadow: bool,
    pub shadow_color: [u8; 3],
    pub shadow_opacity: f32,
    /// Shadow offset in canvas points (right, down).
    pub shadow_offset: [f32; 2],
    /// Shadow softness in canvas points; 0 = hard edge.
    pub shadow_blur: f32,
}
impl Default for TextLayer {
    fn default() -> Self {
        Self { text: "Text".into(), font_path: None, size: 0.06, color: [255; 3], align: TextAlign::Center, pos: [0.5, 0.1],
               outline_width: 0.0, outline_color: [0; 3], shadow: false, shadow_color: [0; 3], shadow_opacity: 0.7,
               shadow_offset: [4.0, 5.0], shadow_blur: 4.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum LayerKind {
    /// The visualizer itself; its look is `ProjectDoc::config`.
    Visualizer,
    Image { media: MediaRef, fit: FitMode, offset: [f32; 2], #[serde(default = "default_scale")] scale: f32 },
    Video { media: MediaRef, fit: FitMode, offset: [f32; 2], #[serde(default = "default_scale")] scale: f32, looping: bool, mute: bool, trim_start_s: f64 },
    Text(TextLayer),
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Layer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub locked: bool,
    pub opacity: f32,
    pub start_s: f64,
    pub end_s: Option<f64>,
    pub fade_in_s: f32,
    pub fade_out_s: f32,
    pub kind: LayerKind,
}

impl Layer {
    pub fn new(id: u64, name: &str, kind: LayerKind) -> Self {
        Self { id, name: name.into(), visible: true, locked: false, opacity: 1.0,
               start_s: 0.0, end_s: None, fade_in_s: 0.0, fade_out_s: 0.0, kind }
    }

    /// Opacity multiplier at time `t` (0 outside the layer's span). The single source of
    /// truth for fades: preview and export must both call this.
    pub fn alpha_at(&self, t: f64, track_len: f64) -> f32 {
        if !self.visible || t < self.start_s { return 0.0; }
        let end = self.end_s.unwrap_or(track_len);
        if t > end { return 0.0; }
        let mut a = self.opacity;
        if self.fade_in_s > 0.0 { a *= (((t - self.start_s) / self.fade_in_s as f64) as f32).clamp(0.0, 1.0); }
        if self.fade_out_s > 0.0 { a *= (((end - t) / self.fade_out_s as f64) as f32).clamp(0.0, 1.0); }
        a
    }
}

/// A track's own visualizer settings. `rev` increases on every edit so change detection does not have
/// to compare whole configurations.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrackLook { pub rev: u32, pub config: AppConfig }

/// One entry of the audio list: a file (audio, or a video whose sound is used) or a silent spacer.
#[derive(Serialize, Deserialize, Clone)]
pub struct AudioItem {
    pub id: u64,
    pub media: MediaRef,
    /// Render only this span (seconds); `None` = whole track.
    #[serde(default)] pub range: Option<(f64, f64)>,
    /// `Some(seconds)`: a silent spacer (no file) of this length.
    #[serde(default, skip_serializing_if = "Option::is_none")] pub silence_s: Option<f64>,
    /// This track's own visualizer settings; `None` = the project-wide look.
    #[serde(default, skip_serializing_if = "Option::is_none")] pub look: Option<TrackLook>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProjectDoc {
    pub version: u32,
    pub name: String,
    pub render: RenderSettings,
    pub config: AppConfig,
    pub layers: Vec<Layer>,
    pub audio: Vec<AudioItem>,
    pub next_id: u64,
}

impl Default for ProjectDoc {
    fn default() -> Self {
        Self {
            version: PROJECT_VERSION,
            name: "Untitled".into(),
            render: RenderSettings::default(),
            config: AppConfig::default(),
            layers: vec![Layer::new(1, "Visualizer", LayerKind::Visualizer)],
            audio: vec![],
            next_id: 2,
        }
    }
}

impl ProjectDoc {
    pub fn alloc_id(&mut self) -> u64 { let i = self.next_id; self.next_id += 1; i }

    /// Atomic save (write temp, then rename) so a crash never leaves a half-written project.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Cannot read project: {e}"))?;
        let mut doc: ProjectDoc = serde_json::from_slice(&bytes).map_err(|e| format!("Project file is damaged: {e}"))?;
        if doc.version > PROJECT_VERSION {
            return Err(format!("Project was saved by a newer version (format {}).", doc.version));
        }
        // Future migrations: `if doc.version < 2 { ...; doc.version = 2; }`
        doc.config.refresh_premade_presets(); // premade presets are not serialized
        for a in doc.audio.iter_mut() { if let Some(l) = &mut a.look { l.config.refresh_premade_presets(); } }
        Ok(doc)
    }

    /// Every media reference in the project (audio, image and video layers).
    pub fn media_refs_mut(&mut self) -> Vec<&mut MediaRef> {
        let mut v: Vec<&mut MediaRef> = Vec::new();
        for a in self.audio.iter_mut().filter(|a| a.silence_s.is_none()) { v.push(&mut a.media); }
        for l in self.layers.iter_mut() {
            match &mut l.kind {
                LayerKind::Image { media, .. } | LayerKind::Video { media, .. } => v.push(media),
                _ => {}
            }
        }
        v
    }

    /// Recompute relative paths for a (new) project location.
    pub fn refresh_rel(&mut self, dir: &Path) {
        for m in self.media_refs_mut() { m.rel = m.path.strip_prefix(dir).ok().map(|p| p.to_path_buf()); }
    }

    /// Point every reference to `old` at `new` (relinking). Returns how many were changed.
    pub fn relink(&mut self, old: &Path, new: &Path) -> usize {
        let mut n = 0;
        for m in self.media_refs_mut() {
            if m.path == old { m.path = new.to_path_buf(); m.size = std::fs::metadata(new).map(|x| x.len()).unwrap_or(0); n += 1; }
        }
        n
    }

    /// Replace each reference's path with its resolved location when one is found.
    pub fn resolve_all(&mut self, dir: &Path) {
        for m in self.media_refs_mut() { if let Some(p) = m.resolve(dir) { m.path = p; } }
    }

    /// Every media file that can no longer be found.
    pub fn missing_media(&self, project_dir: &Path) -> Vec<PathBuf> {
        let mut out = vec![];
        for a in self.audio.iter().filter(|a| a.silence_s.is_none()) {
            if a.media.resolve(project_dir).is_none() { out.push(a.media.path.clone()); }
        }
        for l in &self.layers {
            match &l.kind {
                LayerKind::Image { media, .. } | LayerKind::Video { media, .. } => {
                    if media.resolve(project_dir).is_none() { out.push(media.path.clone()); }
                }
                _ => {}
            }
        }
        out
    }
}

// ── Undo / redo (snapshot based, edit coalescing by explicit commit) ──

pub struct UndoStack<T: Clone> {
    baseline: T,
    past: Vec<T>,
    future: Vec<T>,
    limit: usize,
}

impl<T: Clone> UndoStack<T> {
    pub fn new(initial: T) -> Self { Self { baseline: initial, past: vec![], future: vec![], limit: 60 } }
    /// The last recorded state; compare the live state against it to detect edits.
    pub fn baseline(&self) -> &T { &self.baseline }

    /// Record `current` as a new step (the caller already knows something changed).
    /// Call when an edit FINISHES, not on every frame of a drag.
    pub fn push(&mut self, current: &T) {
        self.past.push(std::mem::replace(&mut self.baseline, current.clone()));
        self.future.clear();
        if self.past.len() > self.limit { self.past.remove(0); }
    }
    pub fn undo(&mut self) -> Option<T> {
        let prev = self.past.pop()?;
        self.future.push(std::mem::replace(&mut self.baseline, prev));
        Some(self.baseline.clone())
    }
    pub fn redo(&mut self) -> Option<T> {
        let next = self.future.pop()?;
        self.past.push(std::mem::replace(&mut self.baseline, next));
        Some(self.baseline.clone())
    }
    pub fn can_undo(&self) -> bool { !self.past.is_empty() }
    pub fn can_redo(&self) -> bool { !self.future.is_empty() }
    /// After loading a project: forget history.
    pub fn reset(&mut self, initial: T) { *self = Self::new(initial); }
}

impl<T: Clone + PartialEq> UndoStack<T> {
    /// `push` only if `current` differs from the baseline.
    pub fn commit(&mut self, current: &T) -> bool {
        if *current == self.baseline { return false; }
        self.push(current);
        true
    }
}

// ── Presets (reusable look: visualizer config + render settings + layers, no audio) ──

#[derive(Serialize, Deserialize, Clone)]
pub struct PresetFile {
    pub version: u32,
    pub name: String,
    pub render: RenderSettings,
    pub config: AppConfig,
    pub layers: Vec<Layer>,
}

impl PresetFile {
    pub fn from_doc(doc: &ProjectDoc, name: &str) -> Self {
        let mut render = doc.render.clone();
        render.ffmpeg_path = None; // machine specific
        Self { version: PROJECT_VERSION, name: name.into(), render, config: doc.config.clone(), layers: doc.layers.clone() }
    }
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let b = serde_json::to_vec_pretty(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, b)
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("Cannot read preset: {e}"))?;
        let mut p: PresetFile = serde_json::from_slice(&b).map_err(|e| format!("Preset file is damaged: {e}"))?;
        p.config.refresh_premade_presets();
        Ok(p)
    }
    /// Apply onto a project; audio and the machine-specific paths (output folder, ffmpeg) are kept.
    pub fn apply_to(&self, doc: &mut ProjectDoc) {
        let (dir, ff) = (doc.render.output_dir.clone(), doc.render.ffmpeg_path.clone());
        doc.render = self.render.clone();
        doc.render.output_dir = dir;
        doc.render.ffmpeg_path = ff;
        doc.config = self.config.clone();
        doc.layers = self.layers.clone();
        for i in 0..doc.layers.len() { doc.layers[i].id = doc.alloc_id(); }
    }
}

// ── Autosave / recovery ──────────────────────────────────────

pub fn recovery_dir() -> PathBuf {
    dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("RustVisualizer").join("recovery")
}

/// Per-render scratch directory; sweep stale ones on startup with `sweep_stale_temp`.
pub fn render_temp_dir(job_id: u64) -> PathBuf {
    std::env::temp_dir().join("sabi-neiro-render").join(format!("job-{job_id}"))
}
pub fn sweep_stale_temp() {
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("sabi-neiro-render"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_redo_roundtrip() {
        let mut u = UndoStack::new(1);
        assert!(!u.commit(&1));
        assert!(u.commit(&2));
        assert!(u.commit(&3));
        assert_eq!(u.undo(), Some(2));
        assert_eq!(u.undo(), Some(1));
        assert_eq!(u.undo(), None);
        assert_eq!(u.redo(), Some(2));
        assert!(u.commit(&9));        // new edit clears redo
        assert!(!u.can_redo());
    }

    #[test]
    fn fade_alpha() {
        let mut l = Layer::new(1, "t", LayerKind::Visualizer);
        l.fade_in_s = 2.0; l.fade_out_s = 2.0; l.end_s = Some(10.0);
        assert_eq!(l.alpha_at(0.0, 10.0), 0.0);
        assert!((l.alpha_at(1.0, 10.0) - 0.5).abs() < 1e-6);
        assert_eq!(l.alpha_at(5.0, 10.0), 1.0);
        assert!((l.alpha_at(9.0, 10.0) - 0.5).abs() < 1e-6);
        assert_eq!(l.alpha_at(11.0, 10.0), 0.0);
    }
}
