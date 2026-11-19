//! Project document: everything that is saved to a project file and that undo/redo covers.
//! Layer order: index 0 is the BOTTOM layer.
//! Requires `PartialEq` on `AppConfig` (see PATCHES.md).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::AppConfig;

pub const PROJECT_VERSION: u32 = 1;

// ── Media references (relinking / portability) ───────────────

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct MediaRef {
    pub path: PathBuf,
    /// Path relative to the project file, when the media lives under it.
    #[serde(default)] pub rel: Option<PathBuf>,
    #[serde(default)] pub size: u64,
}

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

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    /// Opaque fill colour used when the export is not transparent.
    pub background: [u8; 3],
    pub transparent: bool,
}
impl Default for Canvas {
    fn default() -> Self { Self { width: 1920, height: 1080, fps: 60.0, background: [0, 0, 0], transparent: false } }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FitMode { #[default] Cover, Contain, Stretch }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TextAlign { Left, #[default] Center, Right }

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct TextLayer {
    pub text: String,
    pub font_path: Option<String>,
    /// Size as a fraction of canvas height, so text scales with resolution.
    pub size: f32,
    pub color: [u8; 3],
    pub align: TextAlign,
    /// Anchor position, 0..1 of the canvas.
    pub pos: [f32; 2],
}
impl Default for TextLayer {
    fn default() -> Self {
        Self { text: "Text".into(), font_path: None, size: 0.06, color: [255; 3], align: TextAlign::Center, pos: [0.5, 0.1] }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum LayerKind {
    /// The visualizer itself; its look is `ProjectDoc::config`.
    Visualizer,
    Image { media: MediaRef, fit: FitMode, offset: [f32; 2] },
    Video { media: MediaRef, fit: FitMode, offset: [f32; 2], looping: bool, mute: bool, trim_start_s: f64 },
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

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct AudioItem {
    pub id: u64,
    pub media: MediaRef,
    /// Render only this span (seconds); `None` = whole track.
    #[serde(default)] pub range: Option<(f64, f64)>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct ProjectDoc {
    pub version: u32,
    pub name: String,
    pub canvas: Canvas,
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
            canvas: Canvas::default(),
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
        Ok(doc)
    }

    /// Every media file that can no longer be found.
    pub fn missing_media(&self, project_dir: &Path) -> Vec<PathBuf> {
        let mut out = vec![];
        for a in &self.audio {
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

pub struct UndoStack<T: Clone + PartialEq> {
    baseline: T,
    past: Vec<T>,
    future: Vec<T>,
    limit: usize,
}

impl<T: Clone + PartialEq> UndoStack<T> {
    pub fn new(initial: T) -> Self { Self { baseline: initial, past: vec![], future: vec![], limit: 200 } }

    /// Call when an edit FINISHES (button click, slider released, text field lost focus),
    /// not on every frame of a drag. Returns true if a step was recorded.
    pub fn commit(&mut self, current: &T) -> bool {
        if *current == self.baseline { return false; }
        self.past.push(std::mem::replace(&mut self.baseline, current.clone()));
        self.future.clear();
        if self.past.len() > self.limit { self.past.remove(0); }
        true
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
