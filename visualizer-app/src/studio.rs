//! File mode ("studio"): audio list, background decode + analysis, transport and timeline.
//!
//! Pipeline for the selected entry:  decode -> waveform -> analyze (AnalysisTrack) -> FxTrack.
//! Playback is driven by `Player` (sample-counter clock); the visuals for time `t` come from
//! `frame_for(cfg, t)`, which is what makes preview deterministic and seek-safe.
//! Changing audio/reactive settings re-runs only the analysis stage (debounced).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use visualizer_core::audio::offline::{self, analysis_rate_for_fps, AnalysisTrack};
use visualizer_core::audio::BarFrame;
use visualizer_core::config::{AppConfig, AudioSettings, ReactiveSettings};
use visualizer_core::decode::{self, DecodeError, TrackMeta};
use visualizer_core::musicbee::TrackInfo;
use visualizer_core::project::{AudioItem, MediaRef};
use visualizer_core::playback::{Player, Track};
use visualizer_core::waveform::Waveform;

use visualizer_core::project::Layer;

use crate::fxtrack::{FxCursor, FxTrack};
use crate::timeline::{self, Timeline};
use crate::reactive::FxState;

const EXTS: &[&str] = &["mp3", "flac", "wav", "ogg", "m4a", "aac", "mp4"];

#[derive(Clone, Debug)]
pub enum Status { Pending, Loading, Analyzing, Ready, Failed(String) }

pub struct Entry {
    pub id: u64,
    pub path: PathBuf,
    pub meta: TrackMeta,
    pub duration: Option<f32>,
    pub status: Status,
}
impl Entry {
    fn display(&self) -> String {
        if self.meta.artist.is_empty() { self.meta.title.clone() }
        else { format!("{} \u{2013} {}", self.meta.artist, self.meta.title) }
    }
}

struct Built { tr: AnalysisTrack, fx: FxTrack }
struct Loaded { id: u64, track: Arc<Track>, wave: Arc<Waveform> }

enum Msg {
    Picked(Vec<PathBuf>),
    Probed { id: u64, res: Result<(TrackMeta, Option<f32>), String> },
    Decoded { id: u64, gen: u64, res: Result<(Arc<Track>, Arc<Waveform>, TrackMeta), String> },
    Progress { gen: u64, p: f32 },
    Analyzed { gen: u64, built: Arc<Built> },
    Failed { gen: u64, err: String },
}

enum Action { Select(u64), Remove(u64), Move(u64, i32) }

pub struct Studio {
    pub entries: Vec<Entry>,
    next_id: u64,
    pub selected: Option<u64>,
    player: Player,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    gen: u64,
    loaded: Option<Loaded>,
    built: Option<Arc<Built>>,
    cursor: FxCursor,
    analyzing: bool,
    progress: f32,
    want_play: bool,
    pub autoplay_next: bool,
    pub fps: f32,
    seen_audio: Option<AudioSettings>,
    seen_reactive: Option<ReactiveSettings>,
    stale_since: Option<Instant>,
    pending_track_info: Option<TrackInfo>,
    /// Human-readable import/analysis problems (shown in the list; step 3 adds a log file).
    pub log: Vec<String>,
    /// Bumped whenever the list of files changes (add / remove / reorder / project load).
    pub list_gen: u64,
    timeline: Timeline,
    mark_in: bool,
    mark_out: bool,
    last_mouse: Instant,
    cursor_visible: bool,
}

fn fmt_time(s: f32) -> String {
    let s = s.max(0.0);
    format!("{}:{:02}", (s / 60.0) as u32, (s % 60.0) as u32)
}

impl Studio {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self {
            entries: vec![], next_id: 1, selected: None, player: Player::new(),
            tx, rx, cancel: Arc::new(AtomicBool::new(false)), gen: 0,
            loaded: None, built: None, cursor: FxCursor::new(), analyzing: false, progress: 0.0,
            want_play: false, autoplay_next: true, fps: 60.0,
            seen_audio: None, seen_reactive: None, stale_since: None,
            pending_track_info: None, log: vec![], list_gen: 0,
            timeline: Timeline::new(), mark_in: false, mark_out: false, last_mouse: Instant::now(), cursor_visible: true,
        }
    }

    /// True once a file is analysed and its visuals can replace the live input.
    pub fn is_active(&self) -> bool { self.built.is_some() }
    pub fn position_s(&self) -> f32 { self.player.position_s() }
    pub fn take_track_info(&mut self) -> Option<TrackInfo> { self.pending_track_info.take() }

    // ── import / list management ────────────────────────────

    pub fn open_dialog(&self, ctx: &egui::Context) {
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            if let Some(p) = rfd::FileDialog::new().add_filter("Audio", EXTS).pick_files() {
                let _ = tx.send(Msg::Picked(p));
                ctx.request_repaint();
            }
        });
    }

    pub fn add_files(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        for path in paths {
            let ok = path.extension().and_then(|e| e.to_str())
                .map(|e| EXTS.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false);
            if !ok {
                self.log.push(format!("Skipped {}: unsupported file type", path.display()));
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            self.entries.push(Entry { id, path: path.clone(), meta: TrackMeta::default(), duration: None, status: Status::Pending });
            self.list_gen += 1;
            let (tx, ctx2) = (self.tx.clone(), ctx.clone());
            std::thread::spawn(move || {
                let res = decode::probe_file(&path).map_err(|e| e.to_string());
                let _ = tx.send(Msg::Probed { id, res });
                ctx2.request_repaint();
            });
        }
        if self.selected.is_none() {
            if let Some(first) = self.entries.first().map(|e| e.id) { self.select(ctx, first, false); }
        }
    }

    fn remove(&mut self, ctx: &egui::Context, id: u64) {
        let was_selected = self.selected == Some(id);
        self.entries.retain(|e| e.id != id);
        self.list_gen += 1;
        if was_selected {
            self.cancel.store(true, Ordering::Relaxed);
            self.gen += 1;
            self.selected = None; self.loaded = None; self.built = None; self.analyzing = false;
            self.player.set_track(None);
            if let Some(n) = self.entries.first().map(|e| e.id) { self.select(ctx, n, false); }
        }
    }

    fn move_entry(&mut self, id: u64, delta: i32) {
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            let j = (i as i32 + delta).clamp(0, self.entries.len() as i32 - 1) as usize;
            self.entries.swap(i, j);
            self.list_gen += 1;
        }
    }

    // ── selection / decode / analysis ───────────────────────

    pub fn select(&mut self, ctx: &egui::Context, id: u64, play: bool) {
        let Some(e) = self.entries.iter_mut().find(|e| e.id == id) else { return };
        self.cancel.store(true, Ordering::Relaxed);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.gen += 1;
        self.selected = Some(id);
        self.loaded = None; self.built = None; self.analyzing = false;
        self.cursor = FxCursor::new();
        self.want_play = play;
        self.player.set_track(None);
        e.status = Status::Loading;

        let (gen, path, cancel, tx, ctx2) = (self.gen, e.path.clone(), self.cancel.clone(), self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let res = match decode::decode_file(&path, &cancel) {
                Ok(d) => {
                    let wave = Waveform::build(&d.samples, d.channels, d.sample_rate);
                    let track = Arc::new(Track { samples: d.samples, channels: d.channels, sample_rate: d.sample_rate });
                    Ok((track, Arc::new(wave), d.meta))
                }
                Err(DecodeError::Cancelled) => return,
                Err(e) => Err(e.to_string()),
            };
            let _ = tx.send(Msg::Decoded { id, gen, res });
            ctx2.request_repaint();
        });
    }

    fn spawn_analysis(&mut self, ctx: &egui::Context, cfg: &AppConfig) {
        let Some(l) = &self.loaded else { return };
        self.cancel.store(true, Ordering::Relaxed);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.gen += 1;
        self.analyzing = true;
        self.progress = 0.0;
        self.stale_since = None;
        let (gen, track, cfg, cancel, tx, ctx2) =
            (self.gen, l.track.clone(), cfg.clone(), self.cancel.clone(), self.tx.clone(), ctx.clone());
        let rate = analysis_rate_for_fps(self.fps);
        std::thread::spawn(move || {
            let txp = tx.clone();
            let ctxp = ctx2.clone();
            let tr = offline::analyze(&track.samples, track.channels, track.sample_rate, &cfg.audio, rate, &cancel, |p| {
                let _ = txp.send(Msg::Progress { gen, p: p * 0.7 });
                ctxp.request_repaint();
            });
            let Some(tr) = tr else { return };
            let fx = FxTrack::build(&tr, &cfg, |p| {
                let _ = tx.send(Msg::Progress { gen, p: 0.7 + p * 0.3 });
                cancel.load(Ordering::Relaxed)
            });
            let msg = match fx {
                Some(fx) => Msg::Analyzed { gen, built: Arc::new(Built { tr, fx }) },
                None => return,
            };
            let _ = tx.send(msg);
            ctx2.request_repaint();
        });
    }

    /// Call once per frame (before drawing panels).
    pub fn poll(&mut self, ctx: &egui::Context, cfg: &AppConfig) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Picked(p) => self.add_files(ctx, p),
                Msg::Probed { id, res } => {
                    if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
                        match res {
                            Ok((meta, dur)) => { e.meta = meta; e.duration = dur; }
                            Err(err) => {
                                self.log.push(format!("{}: {err}", e.path.display()));
                                e.status = Status::Failed(err);
                            }
                        }
                    }
                }
                Msg::Decoded { id, gen, res } => {
                    if gen != self.gen { continue; }
                    match res {
                        Ok((track, wave, meta)) => {
                            if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
                                e.duration = Some(track.duration_s());
                                e.status = Status::Analyzing;
                                self.pending_track_info = Some(TrackInfo {
                                    title: meta.title.clone(), artist: meta.artist.clone(),
                                    album: meta.album.clone(), cover_art: meta.cover.clone(),
                                });
                                e.meta = meta;
                            }
                            self.player.set_track(Some(track.clone()));
                            self.loaded = Some(Loaded { id, track, wave });
                            self.spawn_analysis(ctx, cfg);
                        }
                        Err(err) => {
                            if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
                                self.log.push(format!("{}: {err}", e.path.display()));
                                e.status = Status::Failed(err);
                            }
                        }
                    }
                }
                Msg::Progress { gen, p } => if gen == self.gen { self.progress = p; },
                Msg::Analyzed { gen, built } => {
                    if gen != self.gen { continue; }
                    self.built = Some(built);
                    self.cursor = FxCursor::new();
                    self.analyzing = false;
                    if let Some(id) = self.selected {
                        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) { e.status = Status::Ready; }
                    }
                    if self.want_play { self.player.play(); self.want_play = false; }
                }
                Msg::Failed { gen, err } => if gen == self.gen { self.log.push(err); self.analyzing = false; },
            }
        }

        // Settings changed -> re-run analysis (debounced; the old result stays on screen meanwhile).
        let changed = self.seen_audio.as_ref() != Some(&cfg.audio) || self.seen_reactive.as_ref() != Some(&cfg.reactive);
        if changed {
            self.seen_audio = Some(cfg.audio.clone());
            self.seen_reactive = Some(cfg.reactive.clone());
            self.stale_since = Some(Instant::now());
        }
        if let (Some(b), true) = (&self.built, !self.analyzing) {
            let stale = b.tr.settings != cfg.audio || !b.fx.is_valid_for(cfg)
                || (b.tr.rate_hz - analysis_rate_for_fps(self.fps)).abs() > 0.01;
            if stale {
                let since = *self.stale_since.get_or_insert_with(Instant::now);
                if since.elapsed() > Duration::from_millis(400) { self.spawn_analysis(ctx, cfg); }
                else { ctx.request_repaint_after(Duration::from_millis(100)); }
            }
        }

        // End of track -> next entry (same code path as clicking it, so sync is identical).
        if self.player.ended() && self.autoplay_next {
            if let Some(i) = self.entries.iter().position(|e| Some(e.id) == self.selected) {
                if let Some(next) = self.entries.get(i + 1).map(|e| e.id) { self.select(ctx, next, true); }
            }
        }
        if self.analyzing || self.player.is_playing() { ctx.request_repaint(); }
    }

    /// Visual state for time `t` (seconds on the audio clock). `None` until analysis is ready.
    pub fn frame_for(&mut self, cfg: &AppConfig, t: f32) -> Option<(BarFrame, FxState)> {
        let b = self.built.clone()?;
        let step = b.tr.step_at(t);
        self.cursor.advance_to(&b.fx, &b.tr, cfg, step);
        Some((b.tr.frames[step].clone(), self.cursor.fx.clone()))
    }

    // ── input ───────────────────────────────────────────────

    pub fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() { self.add_files(ctx, dropped); }
        if ctx.wants_keyboard_input() { return; }

        use egui::{Key, Modifiers as M};
        let (space, l5, r5, l1, r1, home, open, mi, mo) = ctx.input_mut(|i| (
            i.consume_key(M::NONE, Key::Space),
            i.consume_key(M::NONE, Key::ArrowLeft), i.consume_key(M::NONE, Key::ArrowRight),
            i.consume_key(M::SHIFT, Key::ArrowLeft), i.consume_key(M::SHIFT, Key::ArrowRight),
            i.consume_key(M::NONE, Key::Home), i.consume_key(M::COMMAND, Key::I),
            i.consume_key(M::NONE, Key::I), i.consume_key(M::NONE, Key::O),
        ));
        if space { self.player.toggle(); }
        if l5 { self.seek_rel(-5.0); }
        if r5 { self.seek_rel(5.0); }
        if l1 { self.seek_rel(-1.0); }
        if r1 { self.seek_rel(1.0); }
        if home { self.player.seek(0.0); }
        if open { self.open_dialog(ctx); }
        if mi { self.mark_in = true; }
        if mo { self.mark_out = true; }
    }

    fn seek_rel(&self, d: f32) {
        let dur = self.loaded.as_ref().map(|l| l.track.duration_s()).unwrap_or(0.0);
        self.player.seek((self.player.position_s() + d).clamp(0.0, dur));
    }

    // ── UI ──────────────────────────────────────────────────

    pub fn ui_left(&mut self, ctx: &egui::Context) {
        let mut action: Option<Action> = None;
        egui::SidePanel::left("studio_files").default_width(280.0).resizable(true).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Audio");
                if ui.button("\u{FF0B} Add\u{2026}").clicked() { self.open_dialog(ctx); }
            });
            ui.small("Drop files here, or Ctrl+I");
            ui.checkbox(&mut self.autoplay_next, "Play next automatically");
            if let Some(e) = self.player.error() {
                ui.colored_label(egui::Color32::from_rgb(255, 120, 120), format!("Audio output: {e}"));
            }
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for e in &self.entries {
                    ui.horizontal(|ui| {
                        let sel = Some(e.id) == self.selected;
                        let dur = e.duration.map(fmt_time).unwrap_or_else(|| "--:--".into());
                        if ui.selectable_label(sel, format!("{}  [{}]", e.display(), dur)).clicked() {
                            action = Some(Action::Select(e.id));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("\u{2715}").on_hover_text("Remove").clicked() { action = Some(Action::Remove(e.id)); }
                            if ui.small_button("\u{25BC}").clicked() { action = Some(Action::Move(e.id, 1)); }
                            if ui.small_button("\u{25B2}").clicked() { action = Some(Action::Move(e.id, -1)); }
                        });
                    });
                    match &e.status {
                        Status::Loading => { ui.small("Decoding\u{2026}"); }
                        Status::Analyzing => { ui.small(format!("Analysing\u{2026} {:.0}%", self.progress * 100.0)); }
                        Status::Failed(err) => { ui.colored_label(egui::Color32::from_rgb(255, 120, 120), egui::RichText::new(err).small()); }
                        _ => {}
                    }
                }
            });
        });
        match action {
            Some(Action::Select(id)) => self.select(ctx, id, false),
            Some(Action::Remove(id)) => self.remove(ctx, id),
            Some(Action::Move(id, d)) => self.move_entry(id, d),
            None => {}
        }
    }

    /// Transport + timeline. `range` is the render range (I / O keys or drag the markers),
    /// `layers` are shown as draggable bars, `selected_layer` is shared with the layer panel.
    pub fn ui_bottom(&mut self, ctx: &egui::Context, range: &mut Option<(f64, f64)>, layers: &mut [Layer], selected_layer: &mut Option<u64>) {
        let dur = self.loaded.as_ref().map(|l| l.track.duration_s()).unwrap_or(0.0);
        let pos = self.player.position_s();
        let has_track = self.loaded.is_some();
        if dur > 0.0 {
            let p = pos as f64;
            if std::mem::take(&mut self.mark_in) { let b = range.map(|r| r.1).unwrap_or(dur as f64); *range = Some((p.min(b - 0.05).max(0.0), b)); }
            if std::mem::take(&mut self.mark_out) { let a = range.map(|r| r.0).unwrap_or(0.0); *range = Some((a, p.max(a + 0.05).min(dur as f64))); }
        } else { self.mark_in = false; self.mark_out = false; }

        let mut jump: Option<i32> = None;
        let mut seek: Option<f32> = None;
        egui::TopBottomPanel::bottom("studio_transport").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("\u{23EE}").on_hover_text("Previous track").clicked() { jump = Some(-1); }
                let label = if self.player.is_playing() { "\u{23F8} Pause" } else { "\u{25B6} Play" };
                if ui.add_enabled(has_track, egui::Button::new(label)).clicked() { self.player.toggle(); }
                if ui.button("\u{23ED}").on_hover_text("Next track").clicked() { jump = Some(1); }
                ui.monospace(format!("{} / {}", fmt_time(pos), fmt_time(dur)));
                let mut v = self.player.volume();
                ui.label("Vol");
                if ui.add(egui::Slider::new(&mut v, 0.0..=1.0).show_value(false)).changed() { self.player.set_volume(v); }
                ui.separator();
                match *range {
                    Some((a, b)) => {
                        ui.colored_label(egui::Color32::from_rgb(255, 160, 40), format!("Render {} \u{2192} {}  ({})", fmt_time(a as f32), fmt_time(b as f32), fmt_time((b - a) as f32)));
                        if ui.small_button("Clear").clicked() { *range = None; }
                    }
                    None => { ui.small("I / O: set render in / out"); }
                }
                if ui.small_button("Fit").on_hover_text("Show the whole track (Ctrl+wheel zooms)").clicked() { self.timeline.fit(); }
            });
            let out = self.timeline.ui(ui, timeline::In {
                dur, pos, playing: self.player.is_playing(), wave: self.loaded.as_ref().map(|l| &*l.wave),
                range, layers, selected: selected_layer, analysing: if self.analyzing { Some(self.progress) } else { None },
            });
            seek = out.seek;
            ui.add_space(4.0);
        });
        if let Some(t) = seek { self.player.seek(t); }

        if let Some(d) = jump {
            if let Some(i) = self.entries.iter().position(|e| Some(e.id) == self.selected) {
                let j = i as i32 + d;
                if j >= 0 { if let Some(id) = self.entries.get(j as usize).map(|e| e.id) {
                    let play = self.player.is_playing();
                    self.select(ctx, id, play);
                } }
            }
        }
    }

    /// Fullscreen: a floating transport that appears on mouse movement and hides (with the
    /// cursor) after 2.5 s. Returns true when the user asked to leave fullscreen.
    pub fn ui_fullscreen_overlay(&mut self, ctx: &egui::Context) -> bool {
        let moved = ctx.input(|i| i.pointer.delta() != egui::Vec2::ZERO || i.pointer.any_down() || i.pointer.any_click());
        if moved { self.last_mouse = Instant::now(); }
        let active = self.last_mouse.elapsed() < Duration::from_millis(2500);
        if active != self.cursor_visible {
            self.cursor_visible = active;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(active));
        }
        if active { ctx.request_repaint_after(Duration::from_millis(300)); } else { return false; }
        let dur = self.loaded.as_ref().map(|l| l.track.duration_s()).unwrap_or(0.0);
        let pos = self.player.position_s();
        let mut exit = false;
        egui::Area::new(egui::Id::new("fs_bar")).anchor(egui::Align2::CENTER_BOTTOM, [0.0, -28.0]).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button(if self.player.is_playing() { "\u{23F8}" } else { "\u{25B6}" }).clicked() { self.player.toggle(); }
                    ui.monospace(format!("{} / {}", fmt_time(pos), fmt_time(dur)));
                    ui.spacing_mut().slider_width = 360.0;
                    let mut p = pos;
                    if dur > 0.0 && ui.add(egui::Slider::new(&mut p, 0.0..=dur).show_value(false)).changed() { self.player.seek(p); }
                    if ui.button("Exit fullscreen (F11)").clicked() { exit = true; }
                });
            });
        });
        exit
    }

    /// True while a file owns the title/cover (the app should ignore system media-session updates).
    pub fn owns_track_info(&self) -> bool { self.selected.is_some() }
}

// ── accessors used by the render panel and preview ───────────
impl Studio {
    pub fn selected_entry(&self) -> Option<&Entry> {
        self.selected.and_then(|id| self.entries.iter().find(|e| e.id == id))
    }
    pub fn duration_s(&self) -> f32 {
        self.loaded.as_ref().map(|l| l.track.duration_s()).unwrap_or(0.0)
    }
    pub fn selected_meta(&self) -> Option<&TrackMeta> { self.selected_entry().map(|e| &e.meta) }
}

// ── project integration ──────────────────────────────────────
impl Studio {
    /// The file list as project items (call only when `list_gen` changed: it stats every file).
    pub fn audio_items(&self, dir: Option<&std::path::Path>) -> Vec<AudioItem> {
        self.entries.iter().map(|e| AudioItem { id: e.id, media: MediaRef::new(&e.path, dir), range: None }).collect()
    }

    fn spawn_probe(&self, ctx: &egui::Context, id: u64, path: PathBuf) {
        let (tx, ctx2) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let res = decode::probe_file(&path).map_err(|e| e.to_string());
            let _ = tx.send(Msg::Probed { id, res });
            ctx2.request_repaint();
        });
    }

    /// Make the list match a project: keeps entries (and their decoded/analysed state) whose
    /// file is unchanged, adds new ones, drops the rest. Missing files stay listed as failed.
    pub fn set_audio(&mut self, ctx: &egui::Context, items: &[AudioItem], dir: &std::path::Path) {
        let mut old = std::mem::take(&mut self.entries);
        let mut fresh: Vec<Entry> = Vec::with_capacity(items.len());
        for it in items {
            let path = it.media.resolve(dir).unwrap_or_else(|| it.media.path.clone());
            if let Some(pos) = old.iter().position(|e| e.path == path) { fresh.push(old.remove(pos)); continue; }
            let id = self.next_id;
            self.next_id += 1;
            let exists = path.is_file();
            fresh.push(Entry { id, path: path.clone(), meta: TrackMeta::default(), duration: None,
                status: if exists { Status::Pending } else { Status::Failed("File is missing \u{2013} use File \u{2192} Missing files".into()) } });
            if exists { self.spawn_probe(ctx, id, path); }
        }
        let selected_gone = self.selected.map(|s| old.iter().any(|e| e.id == s)).unwrap_or(false);
        self.entries = fresh;
        self.list_gen += 1;
        if selected_gone {
            self.cancel.store(true, Ordering::Relaxed);
            self.gen += 1;
            self.selected = None; self.loaded = None; self.built = None; self.analyzing = false;
            self.player.set_track(None);
        }
        if self.selected.is_none() {
            if let Some(id) = self.entries.iter().find(|e| !matches!(e.status, Status::Failed(_))).map(|e| e.id) { self.select(ctx, id, false); }
        }
    }
}
