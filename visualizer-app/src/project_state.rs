//! Project session: current document, undo/redo recording, save/open/recent, relinking,
//! portable "collect files", presets, autosave + crash recovery, unsaved-changes guard.
//!
//! The app's live state stays where it is (AppConfig in the settings mutex, RenderSettings in the
//! render panel, audio list in `Studio`, layers in `doc.layers`). `tick` compares it with the
//! undo baseline once per frame and records a step when an edit has finished.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use visualizer_core::config::AppConfig;
use visualizer_core::export::RenderSettings;
use visualizer_core::project::*;

enum Done {
    Open(PathBuf),
    SaveAs(PathBuf),
    LoadPreset(PathBuf),
    SavePreset(PathBuf),
    Locate(PathBuf, PathBuf),
    SearchFolder(PathBuf),
    Collected(Result<Vec<(PathBuf, PathBuf)>, String>),
}

#[derive(Clone)]
enum Pending { Close, New, Open(PathBuf) }

pub struct UiOut {
    /// A document the app must push into its live state (config, render panel, audio list).
    pub apply: Option<ProjectDoc>,
    pub toggle_render: bool,
    pub toggle_settings: bool,
}

pub struct ProjectState {
    pub doc: ProjectDoc,
    pub path: Option<PathBuf>,
    pub dirty: bool,
    undo: UndoStack<ProjectDoc>,
    config_fp: u64,
    fp_fn: fn(&AppConfig) -> u64,
    tx: Sender<Done>,
    rx: Receiver<Done>,
    autosave_at: Instant,
    recent: Vec<PathBuf>,
    recovery: Option<PathBuf>,
    toast: Option<(String, Instant)>,
    show_missing: bool,
    pending: Option<Pending>,
    after_save: Option<Pending>,
    close_ok: bool,
    queued_apply: Option<ProjectDoc>,
    last_title: Option<String>,
    pub list_gen_seen: u64,
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }
fn recovery_file() -> PathBuf { recovery_dir().join("autosave.sabi.json") }
fn recovery_meta() -> PathBuf { recovery_dir().join("autosave.origin") }
fn lock_file() -> PathBuf { recovery_dir().join("session.lock") }
fn recent_file() -> PathBuf { dirs::config_dir().unwrap_or_else(|| ".".into()).join("RustVisualizer").join("recent.json") }
fn cmd() -> egui::Modifiers { egui::Modifiers { command: true, ..egui::Modifiers::NONE } }
fn cmd_shift() -> egui::Modifiers { egui::Modifiers { command: true, shift: true, ..egui::Modifiers::NONE } }

impl ProjectState {
    pub fn new(config: AppConfig, render: RenderSettings, fp_fn: fn(&AppConfig) -> u64) -> Self {
        let mut doc = ProjectDoc::default();
        doc.config = config;
        doc.render = render;
        let (tx, rx) = channel();
        let recent: Vec<PathBuf> = std::fs::read(recent_file()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        // A leftover autosave whose session lock stopped beating means the last run crashed.
        let stale_lock = std::fs::read_to_string(lock_file()).ok().and_then(|s| s.trim().parse::<u64>().ok())
            .map(|t| now_secs().saturating_sub(t) > 120).unwrap_or(true);
        let recovery = (recovery_file().exists() && stale_lock).then(recovery_file);
        let fp = fp_fn(&doc.config);
        Self { undo: UndoStack::new(doc.clone()), doc, path: None, dirty: false, config_fp: fp, fp_fn, tx, rx,
               autosave_at: Instant::now() - Duration::from_secs(60), recent, recovery, toast: None, show_missing: false,
               pending: None, after_save: None, close_ok: false, queued_apply: None, last_title: None, list_gen_seen: u64::MAX }
    }

    pub fn dir_opt(&self) -> Option<PathBuf> { self.path.as_ref().and_then(|p| p.parent()).map(|p| p.to_path_buf()) }
    pub fn dir(&self) -> PathBuf { self.dir_opt().unwrap_or_else(|| dirs::document_dir().unwrap_or_else(|| PathBuf::from("."))) }
    pub fn reset_title(&mut self) { self.last_title = None; }
    fn say(&mut self, m: impl Into<String>) { self.toast = Some((m.into(), Instant::now())); }

    // ── change tracking / undo ──────────────────────────────

    /// `config_fp` is `Some(fingerprint)` only on frames where the config generation changed.
    pub fn tick(&mut self, ctx: &egui::Context, config: &AppConfig, config_fp: Option<u64>,
                render: &RenderSettings, audio: &[AudioItem], audio_changed: bool) {
        let base = self.undo.baseline();
        let cfg_changed = config_fp.map(|f| f != self.config_fp).unwrap_or(false);
        let changed = cfg_changed
            || self.doc.layers != base.layers
            || *render != base.render
            || (audio_changed && !same_audio(audio, &base.audio));
        if !changed { return; }
        if ctx.input(|i| i.pointer.any_down()) { return; } // wait until the drag ends
        if let (true, Some(f)) = (cfg_changed, config_fp) { self.doc.config = config.clone(); self.config_fp = f; }
        self.doc.render = render.clone();
        if audio_changed { self.doc.audio = audio.to_vec(); }
        self.undo.push(&self.doc);
        self.dirty = true;
    }

    fn adopt(&mut self, d: ProjectDoc) -> ProjectDoc {
        self.config_fp = (self.fp_fn)(&d.config);
        self.doc = d.clone();
        self.dirty = true;
        d
    }
    pub fn undo(&mut self) -> Option<ProjectDoc> { let d = self.undo.undo()?; Some(self.adopt(d)) }
    pub fn redo(&mut self) -> Option<ProjectDoc> { let d = self.undo.redo()?; Some(self.adopt(d)) }
    fn record(&mut self) -> ProjectDoc {
        self.config_fp = (self.fp_fn)(&self.doc.config);
        self.undo.push(&self.doc);
        self.dirty = true;
        self.doc.clone()
    }

    // ── file operations ─────────────────────────────────────

    fn write(&mut self, path: &Path) -> bool {
        if let Some(d) = path.parent() { self.doc.refresh_rel(d); }
        self.doc.name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_string();
        match self.doc.save(path) {
            Ok(()) => {
                self.path = Some(path.to_path_buf());
                self.dirty = false;
                self.push_recent(path);
                self.say("Saved");
                let _ = std::fs::remove_file(recovery_file());
                true
            }
            Err(e) => { self.say(format!("Save failed: {e}")); false }
        }
    }

    fn push_recent(&mut self, p: &Path) {
        self.recent.retain(|r| r != p);
        self.recent.insert(0, p.to_path_buf());
        self.recent.truncate(8);
        let f = recent_file();
        if let Some(d) = f.parent() { let _ = std::fs::create_dir_all(d); }
        if let Ok(b) = serde_json::to_vec(&self.recent) { let _ = std::fs::write(f, b); }
    }

    pub fn save(&mut self) -> bool {
        match self.path.clone() { Some(p) => self.write(&p), None => { self.save_as_dialog(); false } }
    }
    fn save_as_dialog(&self) {
        let (tx, name, dir) = (self.tx.clone(), self.doc.name.clone(), self.dir());
        std::thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new().add_filter("SABI project", &["sabi"]).set_file_name(format!("{name}.sabi")).set_directory(dir).save_file() {
                let _ = tx.send(Done::SaveAs(p));
            }
        });
    }
    fn open_dialog(&self) {
        let (tx, dir) = (self.tx.clone(), self.dir());
        std::thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new().add_filter("SABI project", &["sabi"]).set_directory(dir).pick_file() { let _ = tx.send(Done::Open(p)); }
        });
    }

    fn load_path(&mut self, p: &Path) -> Option<ProjectDoc> {
        match ProjectDoc::load(p) {
            Ok(mut d) => {
                if let Some(dir) = p.parent() { d.resolve_all(dir); }
                self.config_fp = (self.fp_fn)(&d.config);
                self.undo.reset(d.clone());
                self.doc = d.clone();
                self.path = Some(p.to_path_buf());
                self.dirty = false;
                self.push_recent(p);
                self.show_missing = !self.doc.missing_media(&self.dir()).is_empty();
                Some(d)
            }
            Err(e) => { self.say(e); None }
        }
    }

    fn new_project(&mut self) -> ProjectDoc {
        let mut d = ProjectDoc::default();
        d.config = self.doc.config.clone();
        d.render = self.doc.render.clone();
        d.render.range = None;
        self.config_fp = (self.fp_fn)(&d.config);
        self.undo.reset(d.clone());
        self.doc = d.clone();
        self.path = None;
        self.dirty = false;
        d
    }

    fn perform(&mut self, ctx: &egui::Context, p: Pending) {
        match p {
            Pending::Close => { self.close_ok = true; ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
            Pending::New => { let d = self.new_project(); self.queued_apply = Some(d); }
            Pending::Open(path) => { if let Some(d) = self.load_path(&path) { self.queued_apply = Some(d); } }
        }
    }
    /// Run `p` now, or after the user decides what to do with unsaved changes.
    fn guarded(&mut self, ctx: &egui::Context, p: Pending) { if self.dirty { self.pending = Some(p); } else { self.perform(ctx, p); } }

    fn handle(&mut self, ctx: &egui::Context, d: Done) {
        match d {
            Done::Open(p) => self.guarded(ctx, Pending::Open(p)),
            Done::SaveAs(p) => {
                let p = if p.extension().is_none() { p.with_extension("sabi") } else { p };
                if self.write(&p) { if let Some(a) = self.after_save.take() { self.perform(ctx, a); } }
            }
            Done::LoadPreset(p) => match PresetFile::load(&p) {
                Ok(pr) => { pr.apply_to(&mut self.doc); let d = self.record(); self.queued_apply = Some(d); self.say(format!("Preset '{}' applied", pr.name)); }
                Err(e) => self.say(e),
            },
            Done::SavePreset(p) => {
                let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("preset").to_string();
                match PresetFile::from_doc(&self.doc, &name).save(&p) { Ok(()) => self.say("Preset saved"), Err(e) => self.say(format!("Preset not saved: {e}")) }
            }
            Done::Locate(old, new) => {
                if self.doc.relink(&old, &new) > 0 { let d = self.record(); self.queued_apply = Some(d); }
            }
            Done::SearchFolder(dir) => {
                let pd = self.dir();
                let mut n = 0;
                for m in self.doc.media_refs_mut() { if m.resolve(&pd).is_none() && m.relink_in(&dir) { n += 1; } }
                if n > 0 { let d = self.record(); self.queued_apply = Some(d); }
                self.say(format!("Relinked {n} file(s)"));
            }
            Done::Collected(r) => match r {
                Ok(pairs) => {
                    for (o, n) in pairs { self.doc.relink(&o, &n); }
                    if let Some(p) = self.path.clone() { self.write(&p); }
                    let d = self.record();
                    self.queued_apply = Some(d);
                    self.say("All media copied next to the project");
                }
                Err(e) => self.say(format!("Collect failed: {e}")),
            },
        }
    }

    fn collect_files(&mut self) {
        let Some(path) = self.path.clone() else { self.say("Save the project first"); return; };
        let dir = self.dir();
        let target = dir.join(format!("{}_media", self.doc.name));
        let mut seen = HashSet::new();
        let mut files: Vec<PathBuf> = vec![];
        for m in self.doc.media_refs_mut() {
            if let Some(p) = m.resolve(&dir) { if seen.insert(p.clone()) { files.push(p); } }
        }
        let tx = self.tx.clone();
        self.say("Copying media\u{2026}");
        let _ = path;
        std::thread::spawn(move || {
            let r = (|| -> Result<Vec<(PathBuf, PathBuf)>, String> {
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                let mut out = vec![];
                for f in files {
                    if f.starts_with(&target) { continue; }
                    let name = f.file_name().ok_or("bad file name")?;
                    let mut dest = target.join(name);
                    let mut i = 2;
                    while dest.exists() { dest = target.join(format!("{i}_{}", name.to_string_lossy())); i += 1; }
                    std::fs::copy(&f, &dest).map_err(|e| format!("{}: {e}", f.display()))?;
                    out.push((f, dest));
                }
                Ok(out)
            })();
            let _ = tx.send(Done::Collected(r));
        });
    }

    // ── autosave / recovery ─────────────────────────────────

    /// Call every frame; cheap. Writes a heartbeat lock and, when dirty, an autosave.
    pub fn autosave_tick(&mut self) {
        if self.autosave_at.elapsed() < Duration::from_secs(30) { return; }
        self.autosave_at = Instant::now();
        let _ = std::fs::create_dir_all(recovery_dir());
        let _ = std::fs::write(lock_file(), now_secs().to_string());
        if self.dirty {
            if self.doc.save(&recovery_file()).is_ok() {
                let _ = std::fs::write(recovery_meta(), self.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default());
            }
        }
    }
    /// Clean exit: nothing to recover.
    pub fn shutdown(&mut self) {
        let _ = std::fs::remove_file(recovery_file());
        let _ = std::fs::remove_file(recovery_meta());
        let _ = std::fs::remove_file(lock_file());
        visualizer_core::project::sweep_stale_temp();
    }

    // ── UI ──────────────────────────────────────────────────

    pub fn ui(&mut self, ctx: &egui::Context, render_busy: bool, queue_note: &str) -> UiOut {
        let mut out = UiOut { apply: None, toggle_render: false, toggle_settings: false };
        while let Ok(d) = self.rx.try_recv() { self.handle(ctx, d); }

        // window close with unsaved work
        if ctx.input(|i| i.viewport().close_requested()) && !self.close_ok && self.dirty {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending = Some(Pending::Close);
        }

        // shortcuts
        let (save_as, save, open, new, redo_s, undo, redo_y) = ctx.input_mut(|i| (
            i.consume_key(cmd_shift(), egui::Key::S), i.consume_key(cmd(), egui::Key::S), i.consume_key(cmd(), egui::Key::O),
            i.consume_key(cmd(), egui::Key::N), i.consume_key(cmd_shift(), egui::Key::Z), i.consume_key(cmd(), egui::Key::Z),
            i.consume_key(cmd(), egui::Key::Y)));
        let typing = ctx.wants_keyboard_input();
        if save_as { self.save_as_dialog(); } else if save { self.save(); }
        if open { self.open_dialog(); }
        if new { self.guarded(ctx, Pending::New); }
        if !typing {
            if undo { if let Some(d) = self.undo() { self.queued_apply = Some(d); } }
            if redo_s || redo_y { if let Some(d) = self.redo() { self.queued_apply = Some(d); } }
        }

        // menu bar
        egui::TopBottomPanel::top("studio_top").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New          Ctrl+N").clicked() { self.guarded(ctx, Pending::New); ui.close_menu(); }
                    if ui.button("Open\u{2026}      Ctrl+O").clicked() { self.open_dialog(); ui.close_menu(); }
                    ui.menu_button("Open recent", |ui| {
                        if self.recent.is_empty() { ui.label("(none)"); }
                        for r in self.recent.clone() {
                            if ui.button(r.file_name().and_then(|n| n.to_str()).unwrap_or("?")).on_hover_text(r.display().to_string()).clicked() {
                                let _ = self.tx.send(Done::Open(r)); ui.close_menu();
                            }
                        }
                    });
                    ui.separator();
                    if ui.button("Save          Ctrl+S").clicked() { self.save(); ui.close_menu(); }
                    if ui.button("Save as\u{2026}   Ctrl+Shift+S").clicked() { self.save_as_dialog(); ui.close_menu(); }
                    ui.separator();
                    if ui.button("Collect files next to project").on_hover_text("Copies audio, images and videos into <project>_media so the project can be moved").clicked() { self.collect_files(); ui.close_menu(); }
                    if ui.button("Missing files\u{2026}").clicked() { self.show_missing = true; ui.close_menu(); }
                    ui.separator();
                    if ui.button("Save preset\u{2026}").on_hover_text("Visualizer look + render settings + layers, without audio").clicked() {
                        let (tx, dir) = (self.tx.clone(), self.dir());
                        std::thread::spawn(move || { if let Some(p) = rfd::FileDialog::new().add_filter("SABI preset", &["sabipreset"]).set_directory(dir).save_file() { let _ = tx.send(Done::SavePreset(p.with_extension("sabipreset"))); } });
                        ui.close_menu();
                    }
                    if ui.button("Load preset\u{2026}").clicked() {
                        let (tx, dir) = (self.tx.clone(), self.dir());
                        std::thread::spawn(move || { if let Some(p) = rfd::FileDialog::new().add_filter("SABI preset", &["sabipreset"]).set_directory(dir).pick_file() { let _ = tx.send(Done::LoadPreset(p)); } });
                        ui.close_menu();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.add_enabled(self.undo.can_undo(), egui::Button::new("Undo      Ctrl+Z")).clicked() { if let Some(d) = self.undo() { self.queued_apply = Some(d); } ui.close_menu(); }
                    if ui.add_enabled(self.undo.can_redo(), egui::Button::new("Redo      Ctrl+Y")).clicked() { if let Some(d) = self.redo() { self.queued_apply = Some(d); } ui.close_menu(); }
                });
                ui.separator();
                if ui.button("Render\u{2026}  Ctrl+R").clicked() { out.toggle_render = true; }
                if ui.button("Settings").clicked() { out.toggle_settings = true; }
                if render_busy { ui.label(format!("\u{23F3} {queue_note}")); }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{}{}", if self.dirty { "\u{25CF} " } else { "" }, self.path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("Untitled project")));
                });
            });
        });

        // window title
        let title = format!("{}{} \u{2013} SABI NEIRO", if self.dirty { "\u{25CF} " } else { "" }, if self.doc.name.is_empty() { "Untitled" } else { &self.doc.name });
        if self.last_title.as_deref() != Some(title.as_str()) { ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone())); self.last_title = Some(title); }

        self.dialogs(ctx);
        self.autosave_tick();
        if let Some((m, t)) = &self.toast {
            if t.elapsed() < Duration::from_millis(3000) {
                egui::Area::new(egui::Id::new("toast")).anchor(egui::Align2::RIGHT_BOTTOM, [-14.0, -110.0]).show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| { ui.label(m); });
                });
                ctx.request_repaint_after(Duration::from_millis(300));
            } else { self.toast = None; }
        }
        out.apply = self.queued_apply.take();
        out
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        // crash recovery
        if let Some(p) = self.recovery.clone() {
            egui::Window::new("Recover unsaved work?").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                ui.label("The last session ended unexpectedly. An autosaved copy of your project was found.");
                ui.horizontal(|ui| {
                    if ui.button("Recover").clicked() {
                        match ProjectDoc::load(&p) {
                            Ok(mut d) => {
                                let origin = std::fs::read_to_string(recovery_meta()).ok().filter(|s| !s.is_empty()).map(PathBuf::from);
                                if let Some(dir) = origin.as_ref().and_then(|o| o.parent()) { d.resolve_all(dir); }
                                self.config_fp = (self.fp_fn)(&d.config);
                                self.undo.reset(d.clone());
                                self.doc = d.clone();
                                self.path = origin;
                                self.dirty = true;
                                self.queued_apply = Some(d);
                            }
                            Err(e) => self.say(e),
                        }
                        self.recovery = None;
                    }
                    if ui.button("Discard").clicked() { let _ = std::fs::remove_file(&p); self.recovery = None; }
                });
            });
        }

        // unsaved changes
        if let Some(p) = self.pending.clone() {
            egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                ui.label("Save your changes first?");
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.pending = None;
                        if self.path.is_some() { if self.save() { self.perform(ctx, p.clone()); } }
                        else { self.after_save = Some(p.clone()); self.save_as_dialog(); }
                    }
                    if ui.button("Don't save").clicked() { self.pending = None; self.dirty = false; self.perform(ctx, p.clone()); }
                    if ui.button("Cancel").clicked() { self.pending = None; }
                });
            });
        }

        // missing media
        if self.show_missing {
            let dir = self.dir();
            let mut missing: Vec<PathBuf> = vec![];
            for m in self.doc.media_refs_mut() { if m.resolve(&dir).is_none() && !missing.contains(&m.path) { missing.push(m.path.clone()); } }
            let mut open = true;
            egui::Window::new("Missing files").open(&mut open).default_width(520.0).show(ctx, |ui| {
                if missing.is_empty() { ui.label("All media was found."); return; }
                ui.label("These files could not be found. Locate each one, or point to a folder that contains them.");
                for old in &missing {
                    ui.horizontal(|ui| {
                        ui.small(old.display().to_string());
                        if ui.small_button("Locate\u{2026}").clicked() {
                            let (tx, o) = (self.tx.clone(), old.clone());
                            std::thread::spawn(move || { if let Some(n) = rfd::FileDialog::new().set_file_name(o.file_name().and_then(|n| n.to_str()).unwrap_or("")).pick_file() { let _ = tx.send(Done::Locate(o, n)); } });
                        }
                    });
                }
                if ui.button("Search a folder\u{2026}").clicked() {
                    let tx = self.tx.clone();
                    std::thread::spawn(move || { if let Some(d) = rfd::FileDialog::new().pick_folder() { let _ = tx.send(Done::SearchFolder(d)); } });
                }
            });
            if !open || missing.is_empty() { self.show_missing = false; }
        }
    }
}

fn same_audio(a: &[AudioItem], b: &[AudioItem]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.media.path == y.media.path)
}
