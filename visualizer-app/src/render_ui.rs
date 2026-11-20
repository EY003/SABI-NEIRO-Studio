//! Render panel: settings, preflight messages, queue. Pure egui; all heavy work lives in render.rs.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

use eframe::egui;
use visualizer_core::config::AppConfig;
use visualizer_core::export::*;
use visualizer_core::project::Layer;

use crate::render::{JobState, RenderQueue};
use crate::studio::Studio;

const RESOLUTIONS: &[(&str, u32, u32)] = &[
    ("720p 16:9", 1280, 720), ("1080p 16:9", 1920, 1080), ("1440p 16:9", 2560, 1440), ("4K 16:9", 3840, 2160),
    ("Vertical 9:16 (1080x1920)", 1080, 1920), ("Square 1:1 (1080x1080)", 1080, 1080),
    ("4:3 (1440x1080)", 1440, 1080), ("Ultrawide 21:9 (2560x1080)", 2560, 1080),
];
const FPS_LIST: &[f32] = &[23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0];

pub struct RenderUi {
    pub open: bool,
    pub settings: RenderSettings,
    pub show_guides: bool,
    /// ffmpeg path once discovered (the compositor needs it for video layers).
    saved: RenderSettings,
    caps: Option<FfmpegCaps>,
    caps_err: Option<String>,
    caps_rx: Option<Receiver<Result<FfmpegCaps, String>>>,
    folder_rx: Option<Receiver<PathBuf>>,
    issues: Vec<Issue>,
    checked_at: Option<Instant>,
    checked_for: Option<(RenderSettings, Option<PathBuf>)>,
    log_job: Option<u64>,
}

fn prefs_path() -> PathBuf {
    dirs::config_dir().unwrap_or_else(|| ".".into()).join("RustVisualizer").join("render.toml")
}

pub fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let existing = dir.ancestors().find(|p| p.exists())?;
    let w: Vec<u16> = existing.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut avail = 0u64;
    let ok = unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut avail, std::ptr::null_mut(), std::ptr::null_mut()) };
    (ok != 0).then_some(avail)
}

fn mmss(s: f32) -> String { format!("{}:{:02}", (s / 60.0) as u32, (s % 60.0) as u32) }

impl RenderUi {
    pub fn new() -> Self {
        let settings: RenderSettings = std::fs::read_to_string(prefs_path()).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default();
        Self { open: false, saved: settings.clone(), settings, show_guides: true, caps: None, caps_err: None, caps_rx: None,
               folder_rx: None, issues: vec![], checked_at: None, checked_for: None, log_job: None }
    }

    pub fn ffmpeg_path(&self) -> Option<PathBuf> { self.caps.as_ref().map(|c| c.path.clone()) }

    fn save_if_changed(&mut self) {
        if self.settings != self.saved {
            self.saved = self.settings.clone();
            let p = prefs_path();
            if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
            if let Ok(s) = toml::to_string(&self.settings) { let _ = std::fs::write(p, s); }
        }
    }

    fn start_probe(&mut self) {
        let custom = self.settings.ffmpeg_path.clone();
        let (tx, rx) = channel();
        self.caps_rx = Some(rx);
        std::thread::spawn(move || {
            let r = FfmpegCaps::discover(custom.as_deref()).map(|mut c| { c.probe_hardware(); c });
            let _ = tx.send(r);
        });
    }

    fn fix_container(&mut self) {
        let s = &mut self.settings;
        if !container_video_ok(s.container, s.video_codec) {
            s.video_codec = VideoCodec::ALL.iter().copied().find(|v| container_video_ok(s.container, *v)).unwrap_or(VideoCodec::H264);
        }
        if !container_audio_ok(s.container, s.audio_codec) { s.audio_codec = default_audio_for(s.container); }
        if !hw_supported(s.video_codec, s.hw) { s.hw = HwAccel::None; }
    }

    pub fn ui(&mut self, ctx: &egui::Context, studio: &Studio, cfg: &AppConfig, queue: &mut RenderQueue,
              layers: &[Layer], project_dir: &Path, extra: &[Issue]) {
        if !self.open { self.save_if_changed(); return; }
        if self.caps_rx.is_none() && self.caps.is_none() && self.caps_err.is_none() { self.start_probe(); }
        if let Some(rx) = &self.caps_rx {
            if let Ok(r) = rx.try_recv() {
                match r { Ok(c) => self.caps = Some(c), Err(e) => self.caps_err = Some(e) }
                self.caps_rx = None;
            }
        }
        if let Some(rx) = &self.folder_rx {
            if let Ok(p) = rx.try_recv() { self.settings.output_dir = p; self.folder_rx = None; }
        }
        let mut open = self.open;
        egui::Window::new("Render").open(&mut open).default_width(500.0).default_height(640.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.body(ui, ctx, studio, cfg, queue, layers, project_dir, extra));
        });
        self.open = open;
        self.save_if_changed();
    }

    fn body(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, studio: &Studio, cfg: &AppConfig, queue: &mut RenderQueue,
            layers: &[Layer], project_dir: &Path, extra: &[Issue]) {
        // ── ffmpeg status ──
        if let Some(e) = &self.caps_err {
            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), e);
            ui.horizontal(|ui| {
                let mut txt = self.settings.ffmpeg_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
                if ui.add(egui::TextEdit::singleline(&mut txt).hint_text("path to ffmpeg.exe").desired_width(300.0)).changed() {
                    self.settings.ffmpeg_path = if txt.is_empty() { None } else { Some(PathBuf::from(txt)) };
                }
                if ui.button("Retry").clicked() { self.caps_err = None; self.caps = None; }
            });
        } else if let Some(c) = &self.caps {
            ui.small(format!("{}  ·  GPU encoders: {}", c.version.split(" Copyright").next().unwrap_or(""),
                if c.working.is_empty() { "none detected".to_string() } else { let mut v: Vec<_> = c.working.iter().cloned().collect(); v.sort(); v.join(", ") }));
        } else { ui.small("Looking for ffmpeg\u{2026}"); }
        ui.separator();

        // ── video ──
        egui::Grid::new("render_grid").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            let s = &mut self.settings;
            ui.label("Resolution");
            ui.horizontal(|ui| {
                let cur = RESOLUTIONS.iter().find(|r| r.1 == s.width && r.2 == s.height).map(|r| r.0).unwrap_or("Custom");
                egui::ComboBox::from_id_source("res").selected_text(cur).show_ui(ui, |ui| {
                    for r in RESOLUTIONS { if ui.selectable_label(cur == r.0, r.0).clicked() { s.width = r.1; s.height = r.2; } }
                });
                ui.add(egui::DragValue::new(&mut s.width).clamp_range(64..=7680).speed(2));
                ui.label("\u{00D7}");
                ui.add(egui::DragValue::new(&mut s.height).clamp_range(64..=7680).speed(2));
            });
            ui.end_row();

            ui.label("Frame rate");
            egui::ComboBox::from_id_source("fps").selected_text(format!("{} fps", s.fps)).show_ui(ui, |ui| {
                for f in FPS_LIST { if ui.selectable_label((s.fps - f).abs() < 0.01, format!("{f}")).clicked() { s.fps = *f; } }
            });
            ui.end_row();

            ui.label("Container");
            let before = s.container;
            ui.horizontal(|ui| { for c in Container::ALL { ui.selectable_value(&mut s.container, c, c.label()); } });
            ui.end_row();
            let changed_container = before != s.container;

            ui.label("Video codec");
            egui::ComboBox::from_id_source("vc").selected_text(s.video_codec.label()).show_ui(ui, |ui| {
                for v in VideoCodec::ALL { if container_video_ok(s.container, v) { ui.selectable_value(&mut s.video_codec, v, v.label()); } }
            });
            ui.end_row();

            ui.label("Encoder");
            let caps = self.caps.as_ref();
            egui::ComboBox::from_id_source("hw").selected_text(s.hw.label()).show_ui(ui, |ui| {
                for h in HwAccel::ALL {
                    let ok = caps.map(|c| c.hw_available(s.video_codec, h)).unwrap_or(h == HwAccel::None);
                    if ok { ui.selectable_value(&mut s.hw, h, h.label()); }
                }
            });
            ui.end_row();

            ui.label("Decoding");
            ui.checkbox(&mut s.hw_decode, "GPU-decode background videos");
            ui.end_row();

            ui.label("Quality");
            let fixed = matches!(s.video_codec, VideoCodec::ProRes | VideoCodec::Animation);
            ui.add_enabled_ui(!fixed, |ui| {
                ui.horizontal(|ui| {
                    let mut by_q = matches!(s.rate, RateControl::Quality(_));
                    if ui.radio_value(&mut by_q, true, "Quality").clicked() { s.rate = RateControl::Quality(75); }
                    if ui.radio_value(&mut by_q, false, "Bitrate").clicked() { s.rate = RateControl::Bitrate(12_000); }
                    match &mut s.rate {
                        RateControl::Quality(q) => { ui.add(egui::Slider::new(q, 1..=100)); }
                        RateControl::Bitrate(k) => { ui.add(egui::DragValue::new(k).clamp_range(300..=400_000).suffix(" kbps").speed(50)); }
                    }
                });
            });
            ui.end_row();

            ui.label("Audio");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_source("ac").selected_text(s.audio_codec.label()).show_ui(ui, |ui| {
                    for a in AudioCodec::ALL { if container_audio_ok(s.container, a) { ui.selectable_value(&mut s.audio_codec, a, a.label()); } }
                });
                if s.audio_codec.is_lossy() {
                    egui::ComboBox::from_id_source("ab").selected_text(format!("{} kbps", s.audio_bitrate_kbps)).show_ui(ui, |ui| {
                        for b in [96, 128, 192, 256, 320] { ui.selectable_value(&mut s.audio_bitrate_kbps, b, format!("{b} kbps")); }
                    });
                }
            });
            ui.end_row();

            ui.label("Background");
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.transparent, "Transparent");
                if !s.transparent { ui.color_edit_button_srgb(&mut s.background); }
            });
            ui.end_row();
            if changed_container { /* applied below, outside the borrow */ }
        });
        self.fix_container();
        self.settings.width &= !1;
        self.settings.height &= !1;

        // ── range ──
        let dur = studio.duration_s();
        let pos = studio.position_s() as f64;
        ui.horizontal(|ui| {
            let mut on = self.settings.range.is_some();
            if ui.checkbox(&mut on, "Render only a section").changed() {
                self.settings.range = if on { Some((pos.min(dur as f64), (pos + 15.0).min(dur as f64).max(pos + 1.0))) } else { None };
            }
            if let Some((a, b)) = &mut self.settings.range {
                ui.add(egui::DragValue::new(a).clamp_range(0.0..=dur as f64).speed(0.1).suffix(" s"));
                if ui.small_button("\u{21E4} playhead").clicked() { *a = pos; }
                ui.label("to");
                ui.add(egui::DragValue::new(b).clamp_range(0.0..=dur as f64).speed(0.1).suffix(" s"));
                if ui.small_button("\u{21E5} playhead").clicked() { *b = pos; }
            }
        });
        ui.checkbox(&mut self.show_guides, "Show safe-area guides in preview");

        // ── output ──
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Folder");
            let mut txt = self.settings.output_dir.display().to_string();
            if ui.add(egui::TextEdit::singleline(&mut txt).desired_width(280.0)).changed() { self.settings.output_dir = PathBuf::from(txt); }
            if ui.button("Browse\u{2026}").clicked() {
                let (tx, rx) = channel();
                self.folder_rx = Some(rx);
                let start = self.settings.output_dir.clone();
                std::thread::spawn(move || { if let Some(p) = rfd::FileDialog::new().set_directory(start).pick_folder() { let _ = tx.send(p); } });
            }
        });
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add(egui::TextEdit::singleline(&mut self.settings.name_template).desired_width(220.0));
            egui::ComboBox::from_id_source("coll").selected_text(match self.settings.collision { Collision::AutoNumber => "If exists: number it", Collision::Overwrite => "If exists: overwrite" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.settings.collision, Collision::AutoNumber, "If exists: number it");
                    ui.selectable_value(&mut self.settings.collision, Collision::Overwrite, "If exists: overwrite");
                });
        });
        ui.small("Tokens: {artist} {title} {n}");

        // ── preflight (throttled: it touches the disk) ──
        let entry = studio.selected_entry();
        let audio = entry.map(|e| e.path.clone());
        let key = (self.settings.clone(), audio.clone());
        let stale = self.checked_for.as_ref() != Some(&key) || self.checked_at.map(|t| t.elapsed() > Duration::from_secs(2)).unwrap_or(true);
        if stale {
            self.issues.clear();
            if let (Some(e), Some(a)) = (entry, &audio) {
                let out = output_path(&self.settings, &e.meta.title, &e.meta.artist, 1);
                self.issues = preflight(&PreflightInput { settings: &self.settings, caps: self.caps.as_ref(), audio_path: a,
                    audio_duration_s: e.duration, free_bytes: free_bytes(&self.settings.output_dir), out_path: &out });
            } else {
                self.issues.push(Issue { severity: Severity::Error, message: "Add and select an audio file first.".into() });
            }
            self.checked_for = Some(key);
            self.checked_at = Some(Instant::now());
        }
        let all: Vec<Issue> = self.issues.iter().cloned().chain(extra.iter().cloned()).collect();
        let has_error = all.iter().any(|i| i.severity == Severity::Error);
        let mut quick_fix = None;
        for i in &all {
            let (col, tag) = if i.severity == Severity::Error { (egui::Color32::from_rgb(255, 110, 110), "\u{26D4}") } else { (egui::Color32::from_rgb(240, 190, 80), "\u{26A0}") };
            ui.colored_label(col, format!("{tag} {}", i.message));
        }
        if self.settings.transparent && all.iter().any(|i| i.message.contains("transparency")) {
            ui.horizontal(|ui| {
                if ui.button("Use WebM (VP9 + alpha)").clicked() { quick_fix = Some((Container::Webm, VideoCodec::Vp9)); }
                if ui.button("Use MOV (ProRes 4444)").clicked() { quick_fix = Some((Container::Mov, VideoCodec::ProRes)); }
            });
        }
        if let Some((c, v)) = quick_fix { self.settings.container = c; self.settings.video_codec = v; self.settings.hw = HwAccel::None; self.fix_container(); self.checked_for = None; }
        if let Some(e) = entry {
            let d = self.settings.range.map(|(a, b)| b - a).or(e.duration.map(|x| x as f64)).unwrap_or(0.0);
            ui.small(format!("Estimated size \u{2248} {}", human_bytes(estimate_bytes(&self.settings, d))));
        }

        // ── actions ──
        ui.horizontal(|ui| {
            let ready = !has_error && self.caps.is_some();
            if ui.add_enabled(ready, egui::Button::new("Add to queue")).clicked() {
                if let Some(e) = entry { self.enqueue(queue, cfg, e.meta.title.clone(), e.meta.artist.clone(), e.path.clone(), 1, true, layers, project_dir); }
            }
            if ui.add_enabled(self.caps.is_some() && !studio.entries.is_empty(), egui::Button::new("Queue all tracks")).on_hover_text("Uses the settings above for every audio file (range ignored)").clicked() {
                for (n, e) in studio.entries.iter().enumerate() { self.enqueue(queue, cfg, e.meta.title.clone(), e.meta.artist.clone(), e.path.clone(), n + 1, false, layers, project_dir); }
            }
        });

        // ── queue ──
        ui.separator();
        ui.horizontal(|ui| {
            ui.heading("Queue");
            if ui.small_button("Clear finished").clicked() { queue.clear_finished(); }
        });
        let ids: Vec<u64> = queue.jobs.iter().map(|j| j.id).collect();
        let (mut cancel, mut retry, mut remove, mut mv): (Option<u64>, Option<u64>, Option<u64>, Option<(u64, i32)>) = (None, None, None, None);
        let (mut load_into_panel, mut apply_panel, mut refresh_look): (Option<u64>, Option<u64>, Option<u64>) = (None, None, None);
        for j in &queue.jobs {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.strong(&j.title);
                    ui.label(format!("{}x{} {} {}", j.settings.width, j.settings.height, j.settings.video_codec.label(), j.settings.container.label()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("\u{2715}").clicked() { remove = Some(j.id); }
                        if !j.is_active() {
                            if ui.small_button("\u{25BC}").clicked() { mv = Some((j.id, 1)); }
                            if ui.small_button("\u{25B2}").clicked() { mv = Some((j.id, -1)); }
                        }
                    });
                });
                match &j.state {
                    JobState::Queued => { ui.label("Waiting\u{2026}"); }
                    JobState::Preparing => { ui.add(egui::ProgressBar::new(j.progress).text(format!("Analysing audio {:.0}%", j.progress * 100.0))); }
                    JobState::Rendering | JobState::Finishing => {
                        let eta = j.eta_s.map(|e| format!("  ETA {}", mmss(e))).unwrap_or_default();
                        let label = if j.state == JobState::Finishing { "Finalising file\u{2026}".to_string() } else { format!("{:.0}%{eta}", j.progress * 100.0) };
                        ui.add(egui::ProgressBar::new(j.progress).text(label));
                    }
                    JobState::Done => { ui.colored_label(egui::Color32::from_rgb(110, 210, 130), format!("Done in {} \u{2192} {}", mmss(j.elapsed_s), j.out_path.display())); }
                    JobState::Failed(e) => { ui.colored_label(egui::Color32::from_rgb(255, 110, 110), e); }
                    JobState::Cancelled => { ui.label("Cancelled"); }
                }
                ui.horizontal(|ui| {
                    if matches!(j.state, JobState::Queued | JobState::Preparing | JobState::Rendering) && ui.small_button("Cancel").clicked() { cancel = Some(j.id); }
                    if matches!(j.state, JobState::Failed(_) | JobState::Cancelled) && ui.small_button("Retry").clicked() { retry = Some(j.id); }
                    if j.state == JobState::Done && ui.small_button("Show in folder").clicked() {
                        let _ = std::process::Command::new("explorer").arg(format!("/select,{}", j.out_path.display())).spawn();
                    }
                    if matches!(j.state, JobState::Queued | JobState::Failed(_) | JobState::Cancelled) {
                        if ui.small_button("Settings \u{2192} panel").on_hover_text("Load this job's render settings into the panel").clicked() { load_into_panel = Some(j.id); }
                        if ui.small_button("Panel \u{2192} job").on_hover_text("Replace this job's render settings with the panel's (keeps its time range)").clicked() { apply_panel = Some(j.id); }
                        if ui.small_button("Use current look").on_hover_text("Re-snapshot visualizer settings and layers into this job").clicked() { refresh_look = Some(j.id); }
                    }
                    if ui.small_button(if self.log_job == Some(j.id) { "Hide log" } else { "Log" }).clicked() {
                        self.log_job = if self.log_job == Some(j.id) { None } else { Some(j.id) };
                    }
                });
                if self.log_job == Some(j.id) {
                    egui::ScrollArea::vertical().max_height(120.0).id_source(("log", j.id)).show(ui, |ui| { for l in &j.log { ui.small(l); } });
                    ui.small("Full log: %LOCALAPPDATA%\\RustVisualizer\\logs\\render.log");
                }
            });
        }
        let _ = ids;
        if let Some(id) = cancel { if let Some(j) = queue.jobs.iter_mut().find(|j| j.id == id) { j.cancel(); } }
        if let Some(id) = retry { if let Some(j) = queue.jobs.iter_mut().find(|j| j.id == id) { j.retry(); } }
        if let Some(id) = remove { queue.remove(id); }
        if let Some((id, d)) = mv { queue.move_job(id, d); }
        if let Some(id) = load_into_panel { if let Some(j) = queue.jobs.iter().find(|j| j.id == id) { self.settings = j.settings.clone(); } }
        if let Some(id) = apply_panel {
            if let Some(j) = queue.jobs.iter_mut().find(|j| j.id == id) {
                let range = j.settings.range;
                j.settings = self.settings.clone();
                j.settings.range = range;
                j.out_path = j.out_path.with_extension(j.settings.container.ext());
            }
        }
        if let Some(id) = refresh_look {
            if let Some(j) = queue.jobs.iter_mut().find(|j| j.id == id) { j.cfg = Box::new(cfg.clone()); j.layers = layers.to_vec(); }
        }
        let _ = ctx;
    }

    fn enqueue(&mut self, queue: &mut RenderQueue, cfg: &AppConfig, title: String, artist: String, audio: PathBuf, n: usize, with_range: bool,
               layers: &[Layer], project_dir: &Path) {
        let Some(caps) = &self.caps else { return };
        let mut s = self.settings.clone();
        if !with_range { s.range = None; }
        let out = resolve_collision(output_path(&s, &title, &artist, n), s.collision, &queue.taken_paths());
        let label = if artist.is_empty() { title } else { format!("{artist} \u{2013} {title}") };
        queue.add(label, audio, cfg.clone(), s, caps.path.clone(), out, layers.to_vec(), project_dir.to_path_buf());
    }
}
