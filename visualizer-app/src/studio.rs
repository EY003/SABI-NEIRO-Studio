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
use visualizer_core::project::{AudioItem, MediaRef, TrackLook};
use visualizer_core::playback::{Player, Track};
use visualizer_core::waveform::Waveform;

use visualizer_core::project::Layer;

use crate::fxtrack::{FxCursor, FxTrack};
use crate::timeline::{self, Timeline};
use crate::reactive::FxState;

/// Audio files, plus video files whose sound is analysed (through ffmpeg). Formats symphonia cannot
/// read (opus, wma, ac3 ...) also go through ffmpeg.
const EXTS: &[&str] = &["mp3", "flac", "wav", "ogg", "oga", "m4a", "aac", "opus", "wma", "aif", "aiff", "ac3", "mka",
                        "mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts"];

#[derive(Clone, Debug)]
pub enum Status { Pending, Loading, Analyzing, Ready, Failed(String) }

pub struct Entry {
    pub id: u64,
    pub path: PathBuf,
    pub meta: TrackMeta,
    pub duration: Option<f32>,
    pub status: Status,
    /// A silent spacer between tracks (no file; `duration` is its length).
    pub spacer: bool,
    /// This track's own visualizer settings (`None` = the project-wide look).
    pub look: Option<TrackLook>,
}
impl Entry {
    fn display(&self) -> String {
        if self.spacer { return format!("Spacer ({:.1} s of silence)", self.duration.unwrap_or(0.0)); }
        if self.meta.artist.is_empty() { self.meta.title.clone() }
        else { format!("{} - {}", self.meta.artist, self.meta.title) }
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

enum Action { Select(u64), Remove(u64), Move(u64, i32), SpacerLen(u64, f32) }

/// What the floating bar (shown while the panels are hidden) asked for.
pub enum OverlayAct { ShowPanels, ExitFullscreen }

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
    /// All tracks form ONE timeline (set by the app from the render settings).
    pub sequence: bool,
    /// Path of ffmpeg (set by the app once found): needed for video files and exotic audio formats.
    pub ffmpeg: Option<PathBuf>,
    /// A spacer's length was edited; the loaded silence is rebuilt once the mouse is released.
    spacer_dirty: Option<u64>,
    pending_seek: Option<f32>,
    mark_in: bool,
    mark_out: bool,
    last_mouse: Instant,
    cursor_visible: bool,
}

/// Metadata + length. Video files and formats symphonia cannot read go through ffmpeg; containers that
/// declare no length (VBR mp3 ...) are decoded once to measure it.
fn probe_with_duration(path: &std::path::Path, ffmpeg: Option<&std::path::Path>) -> Result<(TrackMeta, Option<f32>), String> {
    let (meta, dur) = decode::probe_any(path, ffmpeg)?;
    if dur.is_some() { return Ok((meta, dur)); }
    let d = decode::decode_any(path, ffmpeg, &AtomicBool::new(false)).map_err(|e| e.to_string())?;
    Ok((meta, Some(d.duration_s())))
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
            timeline: Timeline::new(), sequence: false, ffmpeg: None, spacer_dirty: None, pending_seek: None, mark_in: false, mark_out: false, last_mouse: Instant::now(), cursor_visible: true,
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
            self.entries.push(Entry { id, path: path.clone(), meta: TrackMeta::default(), duration: None, status: Status::Pending, spacer: false, look: None });
            self.list_gen += 1;
            self.spawn_probe(ctx, id, path);
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

        let (gen, path, cancel, tx, ctx2, ffmpeg) = (self.gen, e.path.clone(), self.cancel.clone(), self.tx.clone(), ctx.clone(), self.ffmpeg.clone());
        let spacer_len = if e.spacer { Some(e.duration.unwrap_or(1.0).max(0.1)) } else { None };
        std::thread::spawn(move || {
            if let Some(len) = spacer_len {
                // 1 kHz mono silence: a few KB per second instead of megabytes
                let sr = 1000u32;
                let samples = vec![0.0f32; (len * sr as f32) as usize + 2];
                let wave = Waveform::build(&samples, 1, sr);
                let track = Arc::new(Track { samples, channels: 1, sample_rate: sr });
                let _ = tx.send(Msg::Decoded { id, gen, res: Ok((track, Arc::new(wave), TrackMeta::default())) });
                ctx2.request_repaint();
                return;
            }
            let res = match decode::decode_any(&path, ffmpeg.as_deref(), &cancel) {
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
        let is_spacer = self.entries.iter().any(|e| e.id == l.id && e.spacer);
        let (gen, track, cfg, cancel, tx, ctx2) =
            (self.gen, l.track.clone(), cfg.clone(), self.cancel.clone(), self.tx.clone(), ctx.clone());
        let rate = analysis_rate_for_fps(self.fps);
        std::thread::spawn(move || {
            let txp = tx.clone();
            let ctxp = ctx2.clone();
            let keep = crate::fxtrack::keep_for(&cfg);
            let tr = if is_spacer { Some(offline::silent(track.duration_s(), rate, &cfg.audio)) } else {
                offline::analyze(&track.samples, track.channels, track.sample_rate, &cfg.audio, rate, keep, &cancel, |p| {
                    let _ = txp.send(Msg::Progress { gen, p: p * 0.7 });
                    ctxp.request_repaint();
                })
            };
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
                                if !e.spacer { e.duration = Some(track.duration_s()); } // a spacer keeps its exact length
                                e.status = Status::Analyzing;
                                self.pending_track_info = Some(TrackInfo {
                                    title: meta.title.clone(), artist: meta.artist.clone(),
                                    album: meta.album.clone(), cover_art: meta.cover.clone(),
                                });
                                e.meta = meta;
                            }
                            self.player.set_track(Some(track.clone()));
                            if let Some(t) = self.pending_seek.take() { self.player.seek(t); }
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
            let stale = b.tr.settings != cfg.audio || b.tr.keep != crate::fxtrack::keep_for(cfg) || !b.fx.is_valid_for(cfg)
                || (b.tr.rate_hz - analysis_rate_for_fps(self.fps)).abs() > 0.01;
            if stale {
                let since = *self.stale_since.get_or_insert_with(Instant::now);
                if since.elapsed() > Duration::from_millis(400) { self.spawn_analysis(ctx, cfg); }
                else { ctx.request_repaint_after(Duration::from_millis(100)); }
            }
        }

        if let Some(id) = self.spacer_dirty {
            if !ctx.input(|i| i.pointer.any_down()) {
                self.spacer_dirty = None;
                if self.selected == Some(id) { let play = self.player.is_playing(); self.select(ctx, id, play); }
            }
        }

        // End of track -> next entry (same code path as clicking it, so sync is identical).
        if self.player.ended() && (self.autoplay_next || self.sequence) {
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
        if l5 { self.seek_rel(ctx, -5.0); }
        if r5 { self.seek_rel(ctx, 5.0); }
        if l1 { self.seek_rel(ctx, -1.0); }
        if r1 { self.seek_rel(ctx, 1.0); }
        if home { self.seek_global(ctx, 0.0); }
        if open { self.open_dialog(ctx); }
        if mi { self.mark_in = true; }
        if mo { self.mark_out = true; }
    }

    fn seek_rel(&mut self, ctx: &egui::Context, d: f32) {
        let t = (self.global_pos_s() + d).clamp(0.0, self.total_duration_s());
        self.seek_global(ctx, t);
    }

    // ── one timeline over several tracks ────────────────────

    fn sel_index(&self) -> Option<usize> { self.entries.iter().position(|e| Some(e.id) == self.selected) }

    /// Start of every track on the sequence timeline (+ the total at the end). Unknown lengths count as 0.
    pub fn offsets(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.entries.len() + 1);
        let mut acc = 0.0;
        v.push(0.0);
        for e in &self.entries { acc += e.duration.unwrap_or(0.0); v.push(acc); }
        v
    }
    pub fn total_duration_s(&self) -> f32 {
        if self.sequence { self.offsets().last().copied().unwrap_or(0.0) } else { self.duration_s() }
    }
    /// Playhead on the whole timeline (equals `position_s` when not in sequence mode).
    pub fn global_pos_s(&self) -> f32 {
        let local = self.player.position_s();
        if self.sequence { self.sel_index().map(|i| self.offsets()[i]).unwrap_or(0.0) + local } else { local }
    }
    /// Seek on the whole timeline; switches track when `t` lies in another one.
    pub fn seek_global(&mut self, ctx: &egui::Context, t: f32) {
        if !self.sequence { self.player.seek(t); return; }
        let offs = self.offsets();
        let n = self.entries.len();
        if n == 0 { return; }
        let j = (0..n).rev().find(|&j| offs[j] <= t).unwrap_or(0);
        let local = (t - offs[j]).max(0.0);
        if Some(j) == self.sel_index() { self.player.seek(local); }
        else {
            let play = self.player.is_playing();
            self.pending_seek = Some(local);
            let id = self.entries[j].id;
            self.select(ctx, id, play);
        }
    }
    /// All tracks as export segments; `None` while a length is unknown or a file failed.
    pub fn sequence_segments(&self) -> Option<Vec<visualizer_core::export::AudioSeg>> {
        if self.entries.is_empty() { return None; }
        let offs = self.offsets();
        let mut v = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            if matches!(e.status, Status::Failed(_)) { return None; }
            let d = e.duration?;
            v.push(visualizer_core::export::AudioSeg { path: e.path.clone(), start_s: offs[i] as f64, dur_s: d as f64, silent: e.spacer });
        }
        Some(v)
    }

    // ── UI ──────────────────────────────────────────────────

    /// Sidebar "Audio" tab. Rows are clipped, so a long song name can never widen the sidebar.
    pub fn ui_audio(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, sequence: &mut bool) {
        let mut action: Option<Action> = None;
        ui.horizontal(|ui| {
            if ui.button("+ Add audio / video...").clicked() { self.open_dialog(ctx); }
            ui.small("or drop files / Ctrl+I");
        });
        ui.small("A video's sound can drive the visualizer (needs ffmpeg).");
        ui.checkbox(sequence, "One timeline: all tracks back to back")
            .on_hover_text("Off: every track is its own timeline and its own video.\nOn: tracks are placed one after another; playback, the timeline and the export treat them as ONE long video.");
        let mut add_spacer: Option<f32> = None;
        if *sequence {
            ui.horizontal_wrapped(|ui| {
                ui.label("Add a silent spacer:");
                for s in [1.0_f32, 2.0, 5.0, 10.0] {
                    if ui.small_button(format!("{s:.0} s")).on_hover_text("Silence between tracks, inserted after the selected one. Change its length in the list or by dragging its right edge on the timeline.").clicked() { add_spacer = Some(s); }
                }
            });
        } else {
            ui.checkbox(&mut self.autoplay_next, "Play next track automatically");
        }
        if let Some(e) = self.player.error() {
            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), format!("Audio output: {e}"));
        }
        for l in self.log.iter().rev().take(3) {
            let w = ui.available_width();
            crate::ui_util::clipped_label(ui, l, w, true, Some(egui::Color32::from_rgb(255, 170, 90)));
        }
        ui.separator();

        let offs = self.offsets();
        let seq = *sequence;
        for (i, e) in self.entries.iter().enumerate() {
            ui.horizontal(|ui| {
                let sel = Some(e.id) == self.selected;
                let dur = e.duration.map(fmt_time).unwrap_or_else(|| "--:--".into());
                let own = if e.look.is_some() { "* " } else { "" }; // * = has its own visualizer settings
                let label = if seq { format!("{}  {}{}  [{}]", fmt_time(offs[i]), own, e.display(), dur) } else { format!("{}{}  [{}]", own, e.display(), dur) };
                let reserve = if e.spacer { 76.0 + 78.0 } else { 76.0 };
                let width = (ui.available_width() - reserve).max(40.0);
                if crate::ui_util::clipped_selectable(ui, sel, &label, width).clicked() { action = Some(Action::Select(e.id)); }
                if e.spacer {
                    let mut len = e.duration.unwrap_or(1.0);
                    let r = crate::ui_util::drag_reset(ui, &mut len, 0.1..=600.0, 0.05, "", " s", 2.0);
                    if r.changed() || r.secondary_clicked() { action = Some(Action::SpacerLen(e.id, len)); }
                }
                if ui.small_button("x").on_hover_text("Remove from the list").clicked() { action = Some(Action::Remove(e.id)); }
                if ui.small_button("v").on_hover_text("Move down").clicked() { action = Some(Action::Move(e.id, 1)); }
                if ui.small_button("^").on_hover_text("Move up").clicked() { action = Some(Action::Move(e.id, -1)); }
            });
            match &e.status {
                Status::Loading => { ui.small("   decoding..."); }
                Status::Analyzing => { ui.small(format!("   analysing... {:.0}%", self.progress * 100.0)); }
                Status::Failed(err) => {
                    let w = ui.available_width();
                    crate::ui_util::clipped_label(ui, err, w, true, Some(egui::Color32::from_rgb(255, 110, 110)));
                }
                _ => {}
            }
        }
        match action {
            Some(Action::Select(id)) => self.select(ctx, id, false),
            Some(Action::Remove(id)) => self.remove(ctx, id),
            Some(Action::Move(id, d)) => self.move_entry(id, d),
            Some(Action::SpacerLen(id, len)) => self.set_spacer_len(id, len),
            None => {}
        }
        if let Some(sec) = add_spacer { self.add_spacer(sec); }
        if self.entries.iter().any(|e| e.look.is_some()) { ui.small("* = this track has its own visualizer settings"); }
    }

    /// Transport + timeline. The panel is resizable (drag its top edge); the waveform fills the height.
    pub fn ui_bottom(&mut self, ctx: &egui::Context, range: &mut Option<(f64, f64)>, layers: &mut [Layer], selected_layer: &mut Option<u64>) {
        let dur = self.total_duration_s();
        let pos = self.global_pos_s();
        let has_track = self.loaded.is_some();
        if dur > 0.0 {
            let p = pos as f64;
            if std::mem::take(&mut self.mark_in) { let b = range.map(|r| r.1).unwrap_or(dur as f64); *range = Some((p.min(b - 0.05).max(0.0), b)); }
            if std::mem::take(&mut self.mark_out) { let a = range.map(|r| r.0).unwrap_or(0.0); *range = Some((a, p.max(a + 0.05).min(dur as f64))); }
        } else { self.mark_in = false; self.mark_out = false; }

        let offs = self.offsets();
        let segs: Vec<timeline::Seg> = if self.sequence {
            self.entries.iter().enumerate().map(|(i, e)| timeline::Seg { start: offs[i], len: e.duration.unwrap_or(0.0), name: e.display(), current: Some(e.id) == self.selected, spacer: e.spacer }).collect()
        } else { Vec::new() };
        let wave_off = if self.sequence { self.sel_index().map(|i| offs[i]).unwrap_or(0.0) } else { 0.0 };

        let mut jump: Option<i32> = None;
        let mut seek: Option<f32> = None;
        let mut spacer_edit: Option<(usize, f32)> = None;
        egui::TopBottomPanel::bottom("studio_transport").resizable(true).default_height(270.0).height_range(150.0..=900.0).show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Prev").on_hover_text("Previous track").clicked() { jump = Some(-1); }
                let label = if self.player.is_playing() { "Pause" } else { "Play" };
                if ui.add_enabled(has_track, egui::Button::new(label)).clicked() { self.player.toggle(); }
                if ui.button("Next").on_hover_text("Next track").clicked() { jump = Some(1); }
                ui.monospace(format!("{} / {}", fmt_time(pos), fmt_time(dur)));
                let mut v = self.player.volume();
                ui.label("Vol");
                let vr = ui.add(egui::Slider::new(&mut v, 0.0..=1.0).show_value(false)).on_hover_text("Right-click to reset to 100%");
                if vr.secondary_clicked() { self.player.set_volume(1.0); } else if vr.changed() { self.player.set_volume(v); }
                ui.separator();
                match *range {
                    Some((a, b)) => {
                        ui.colored_label(egui::Color32::from_rgb(255, 160, 40), format!("Render {} - {}  ({})", fmt_time(a as f32), fmt_time(b as f32), fmt_time((b - a) as f32)));
                        if ui.small_button("Zoom to it").clicked() { self.timeline.zoom_range(a as f32, b as f32, dur.max(0.5)); }
                        if ui.small_button("Clear").clicked() { *range = None; }
                    }
                    None => { ui.small("I / O: set render in / out"); }
                }
                ui.separator();
                if ui.small_button("-").on_hover_text("Zoom out").clicked() { self.timeline.zoom_step(1.0 / 1.6, pos, dur.max(0.5)); }
                if ui.small_button("+").on_hover_text("Zoom in").clicked() { self.timeline.zoom_step(1.6, pos, dur.max(0.5)); }
                if ui.small_button("Fit").on_hover_text("Show the whole timeline").clicked() { self.timeline.fit(); }
            });
            let out = self.timeline.ui(ui, timeline::In {
                dur, pos, playing: self.player.is_playing(),
                wave: self.loaded.as_ref().map(|l| (wave_off, &*l.wave)),
                segs: &segs, range, layers, selected: selected_layer,
                analysing: if self.analyzing { Some(self.progress) } else { None },
            });
            seek = out.seek;
            spacer_edit = out.spacer;
            // Fill the panel exactly. If the content were even a pixel shorter, egui would shrink the
            // panel to the content every frame (the "slowly falls back" effect).
            let rest = ui.available_height();
            if rest > 0.0 { ui.allocate_space(egui::vec2(1.0, rest)); }
        });
        if let Some((i, len)) = spacer_edit {
            if let Some(id) = self.entries.get(i).filter(|e| e.spacer).map(|e| e.id) { self.set_spacer_len(id, len); }
        }
        if let Some(t) = seek { self.seek_global(ctx, t); }

        if let Some(d) = jump {
            if let Some(i) = self.sel_index() {
                let j = i as i32 + d;
                if j >= 0 { if let Some(id) = self.entries.get(j as usize).map(|e| e.id) {
                    let play = self.player.is_playing();
                    self.select(ctx, id, play);
                } }
            }
        }
    }

    /// Floating transport shown while the panels are hidden: appears on mouse movement, hides (with the
    /// cursor) after 2.5 s.
    pub fn ui_overlay(&mut self, ctx: &egui::Context, fullscreen: bool) -> Option<OverlayAct> {
        let moved = ctx.input(|i| i.pointer.delta() != egui::Vec2::ZERO || i.pointer.any_down() || i.pointer.any_click());
        if moved { self.last_mouse = Instant::now(); }
        let active = self.last_mouse.elapsed() < Duration::from_millis(2500);
        if fullscreen && active != self.cursor_visible {
            self.cursor_visible = active;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(active));
        }
        if active { ctx.request_repaint_after(Duration::from_millis(300)); } else { return None; }
        let dur = self.total_duration_s();
        let pos = self.global_pos_s();
        let mut act = None;
        egui::Area::new(egui::Id::new("hidden_ui_bar")).anchor(egui::Align2::CENTER_BOTTOM, [0.0, -28.0]).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button(if self.player.is_playing() { "Pause" } else { "Play" }).clicked() { self.player.toggle(); }
                    ui.monospace(format!("{} / {}", fmt_time(pos), fmt_time(dur)));
                    ui.spacing_mut().slider_width = 360.0;
                    let mut p = pos;
                    if dur > 0.0 && ui.add(egui::Slider::new(&mut p, 0.0..=dur).show_value(false)).changed() { self.seek_global(ctx, p); }
                    if ui.button("Show panels (Tab)").clicked() { act = Some(OverlayAct::ShowPanels); }
                    if fullscreen && ui.button("Exit fullscreen (F11)").clicked() { act = Some(OverlayAct::ExitFullscreen); }
                });
            });
        });
        act
    }

    pub fn is_playing(&self) -> bool { self.player.is_playing() }

    /// Free everything heavy (decoded audio, analysis frames, waveform, FX checkpoints). The track list
    /// and the metadata stay, so `reactivate` can bring the selected track back.
    pub fn release(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.cancel = Arc::new(AtomicBool::new(false));
        self.gen += 1;
        self.loaded = None;
        self.built = None;
        self.analyzing = false;
        self.progress = 0.0;
        self.cursor = FxCursor::new();
        self.pending_seek = None;
        self.want_play = false;
        self.player.set_track(None);
        for e in self.entries.iter_mut() {
            if matches!(e.status, Status::Loading | Status::Analyzing | Status::Ready) { e.status = Status::Pending; }
        }
        self.timeline.fit();
    }

    /// Reload the selected track after `release` (decodes and analyses it again).
    pub fn reactivate(&mut self, ctx: &egui::Context) {
        if self.loaded.is_none() { if let Some(id) = self.selected { self.select(ctx, id, false); } }
    }

    // ── spacers ──────────────────────────────────────────────

    /// Insert a silent spacer after the selected entry (or at the end).
    pub fn add_spacer(&mut self, secs: f32) {
        let id = self.next_id;
        self.next_id += 1;
        let at = self.sel_index().map(|i| i + 1).unwrap_or(self.entries.len());
        self.entries.insert(at, Entry { id, path: PathBuf::new(), meta: TrackMeta { title: "Spacer".into(), ..Default::default() },
            duration: Some(secs.max(0.1)), status: Status::Pending, spacer: true, look: None });
        self.list_gen += 1;
    }

    pub fn set_spacer_len(&mut self, id: u64, secs: f32) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id && e.spacer) {
            e.duration = Some(secs.clamp(0.1, 600.0));
            self.list_gen += 1;
            self.spacer_dirty = Some(id);
        }
    }

    // ── per-track visualizer settings ────────────────────────

    pub fn selected_index(&self) -> Option<usize> { self.sel_index() }
    pub fn selected_look(&self) -> Option<&TrackLook> { self.sel_index().and_then(|i| self.entries[i].look.as_ref()) }
    pub fn selected_is_spacer(&self) -> bool { self.sel_index().map(|i| self.entries[i].spacer).unwrap_or(false) }

    /// Give the selected track its own settings (`Some`), or hand it back to the project look (`None`).
    pub fn set_selected_look(&mut self, cfg: Option<AppConfig>) {
        if let Some(i) = self.sel_index() {
            self.entries[i].look = cfg.map(|c| TrackLook { rev: 1, config: c });
            self.list_gen += 1;
        }
    }
    /// The live settings changed while a track with its own settings is selected: store them in that track.
    pub fn store_selected_look(&mut self, cfg: &AppConfig) {
        if let Some(i) = self.sel_index() {
            if let Some(l) = self.entries[i].look.as_mut() { l.config = cfg.clone(); l.rev = l.rev.wrapping_add(1); self.list_gen += 1; }
        }
    }

    /// ffmpeg became available: entries that failed only because it was missing get another try.
    pub fn set_ffmpeg(&mut self, ctx: &egui::Context, ffmpeg: Option<PathBuf>) {
        if self.ffmpeg == ffmpeg { return; }
        self.ffmpeg = ffmpeg;
        if self.ffmpeg.is_none() { return; }
        let retry: Vec<(u64, PathBuf)> = self.entries.iter()
            .filter(|e| !e.spacer && matches!(&e.status, Status::Failed(m) if m.contains("ffmpeg"))).map(|e| (e.id, e.path.clone())).collect();
        for (id, path) in retry {
            if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) { e.status = Status::Pending; }
            self.spawn_probe(ctx, id, path);
        }
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
        self.entries.iter().map(|e| {
            if e.spacer {
                AudioItem { id: e.id, media: MediaRef { path: PathBuf::new(), rel: None, size: 0 }, range: None,
                            silence_s: e.duration.map(|d| d as f64), look: e.look.clone() }
            } else {
                AudioItem { id: e.id, media: MediaRef::new(&e.path, dir), range: None, silence_s: None, look: e.look.clone() }
            }
        }).collect()
    }

    fn spawn_probe(&self, ctx: &egui::Context, id: u64, path: PathBuf) {
        let (tx, ctx2, ff) = (self.tx.clone(), ctx.clone(), self.ffmpeg.clone());
        std::thread::spawn(move || {
            let res = probe_with_duration(&path, ff.as_deref());
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
            if let Some(len) = it.silence_s {
                // spacers have no file: reuse one of the same length, otherwise make a new one
                if let Some(pos) = old.iter().position(|e| e.spacer && (e.duration.unwrap_or(0.0) - len as f32).abs() < 1e-3) {
                    let mut e = old.remove(pos);
                    e.look = it.look.clone();
                    fresh.push(e);
                } else {
                    let id = self.next_id;
                    self.next_id += 1;
                    fresh.push(Entry { id, path: PathBuf::new(), meta: TrackMeta { title: "Spacer".into(), ..Default::default() },
                        duration: Some(len as f32), status: Status::Pending, spacer: true, look: it.look.clone() });
                }
                continue;
            }
            let path = it.media.resolve(dir).unwrap_or_else(|| it.media.path.clone());
            if let Some(pos) = old.iter().position(|e| e.path == path && !e.spacer) {
                let mut e = old.remove(pos);
                e.look = it.look.clone();
                fresh.push(e);
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            let exists = path.is_file();
            fresh.push(Entry { id, path: path.clone(), meta: TrackMeta::default(), duration: None,
                status: if exists { Status::Pending } else { Status::Failed("File is missing - use File > Missing files".into()) },
                spacer: false, look: it.look.clone() });
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
