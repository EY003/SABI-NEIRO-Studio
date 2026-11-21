//! Layer panel (a sidebar tab): stack order, visibility, lock, opacity, timing/fades and per-kind
//! properties. Edits `ProjectDoc::layers` in place; `ProjectState::tick` notices and records undo.
//! Layer 0 is the BOTTOM of the stack; the list shows the top layer first.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use eframe::egui;
use visualizer_core::project::*;

use crate::fonts::{self, SysFont};
use crate::ui_util;

#[derive(Clone, Copy, PartialEq)]
enum Pick { Image, Video, Font(u64), Replace(u64) }

#[derive(Clone, Copy, PartialEq)]
pub enum MediaKind { Image, Video }

pub fn media_kind(path: &Path) -> Option<MediaKind> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "bmp" | "webp" | "gif" | "tif" | "tiff" => Some(MediaKind::Image),
        "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v" | "wmv" | "flv" => Some(MediaKind::Video),
        _ => None,
    }
}

pub struct LayersUi {
    pub selected: Option<u64>,
    pick: Option<(Pick, Receiver<PathBuf>)>,
    sys_fonts: Vec<SysFont>,
    fonts_rx: Option<Receiver<Vec<SysFont>>>,
    fonts_requested: bool,
    font_filter: String,
}

fn kind_label(k: &LayerKind) -> &'static str {
    match k { LayerKind::Visualizer => "visualizer", LayerKind::Image { .. } => "image", LayerKind::Video { .. } => "video", LayerKind::Text(_) => "text" }
}

impl LayersUi {
    pub fn new() -> Self {
        Self { selected: None, pick: None, sys_fonts: vec![], fonts_rx: None, fonts_requested: false, font_filter: String::new() }
    }

    fn start_pick(&mut self, what: Pick) {
        let (tx, rx) = channel();
        self.pick = Some((what, rx));
        std::thread::spawn(move || {
            let d = rfd::FileDialog::new();
            let d = match what {
                Pick::Image => d.add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "gif"]),
                Pick::Video => d.add_filter("Video", &["mp4", "mov", "mkv", "webm", "avi", "m4v"]),
                Pick::Font(_) => d.add_filter("Fonts", &["ttf", "otf", "ttc"]),
                Pick::Replace(_) => d,
            };
            if let Some(p) = d.pick_file() { let _ = tx.send(p); }
        });
    }

    /// Backgrounds go just below the visualizer so the bars stay on top.
    fn visualizer_index(doc: &ProjectDoc) -> usize {
        doc.layers.iter().position(|l| matches!(l.kind, LayerKind::Visualizer)).unwrap_or(0)
    }

    /// Add an image or video layer for `path` (used by the buttons AND by drag-and-drop). False if not media.
    pub fn add_media(&mut self, doc: &mut ProjectDoc, path: &Path, dir: Option<&Path>) -> bool {
        let Some(kind) = media_kind(path) else { return false };
        let id = doc.alloc_id();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Layer").to_string();
        let media = MediaRef::new(path, dir);
        let k = match kind {
            MediaKind::Image => LayerKind::Image { media, fit: FitMode::Cover, offset: [0.0, 0.0], scale: 1.0 },
            MediaKind::Video => LayerKind::Video { media, fit: FitMode::Cover, offset: [0.0, 0.0], scale: 1.0, looping: true, mute: true, trim_start_s: 0.0 },
        };
        let at = Self::visualizer_index(doc);
        doc.layers.insert(at, Layer::new(id, &name, k));
        self.selected = Some(id);
        true
    }

    /// Call once per frame: collects finished file dialogs and the font list.
    pub fn poll(&mut self, doc: &mut ProjectDoc, dir: Option<&Path>) {
        if let Some(rx) = &self.fonts_rx {
            if let Ok(v) = rx.try_recv() { self.sys_fonts = v; self.fonts_rx = None; }
        }
        let Some((what, rx)) = &self.pick else { return };
        let Ok(p) = rx.try_recv() else { return };
        let what = *what;
        self.pick = None;
        match what {
            Pick::Image | Pick::Video => { self.add_media(doc, &p, dir); }
            Pick::Font(id) => if let Some(l) = doc.layers.iter_mut().find(|l| l.id == id) {
                if let LayerKind::Text(t) = &mut l.kind { t.font_path = Some(p.to_string_lossy().into_owned()); }
            },
            Pick::Replace(id) => if let Some(l) = doc.layers.iter_mut().find(|l| l.id == id) {
                match &mut l.kind {
                    LayerKind::Image { media, .. } | LayerKind::Video { media, .. } => *media = MediaRef::new(&p, dir),
                    _ => {}
                }
            },
        }
    }

    fn request_fonts(&mut self) {
        if self.fonts_requested { return; }
        self.fonts_requested = true;
        let (tx, rx) = channel();
        self.fonts_rx = Some(rx);
        std::thread::spawn(move || { let _ = tx.send(fonts::list_system_fonts()); });
    }

    pub fn ui_panel(&mut self, ui: &mut egui::Ui, doc: &mut ProjectDoc, dir: Option<&Path>, playhead: f64) {
        let _ = dir;
        ui.horizontal_wrapped(|ui| {
            if ui.button("+ Image").clicked() { self.start_pick(Pick::Image); }
            if ui.button("+ Video").clicked() { self.start_pick(Pick::Video); }
            if ui.button("+ Text").clicked() {
                let id = doc.alloc_id();
                doc.layers.push(Layer::new(id, "Text", LayerKind::Text(TextLayer::default())));
                self.selected = Some(id);
            }
        });
        ui.small("You can also drop image / video files onto the window.");
        ui.separator();

        // ── stack (top layer first) ──
        let n = doc.layers.len();
        let mut swap: Option<(usize, usize)> = None;
        for idx in (0..n).rev() {
            let l = &mut doc.layers[idx];
            ui.horizontal(|ui| {
                ui.checkbox(&mut l.visible, "").on_hover_text("Visible");
                ui.toggle_value(&mut l.locked, "L").on_hover_text("Lock: blocks edits, moving and deleting");
                let width = (ui.available_width() - 58.0).max(40.0);
                let label = format!("{}  ({})", l.name, kind_label(&l.kind));
                if ui_util::clipped_selectable(ui, self.selected == Some(l.id), &label, width).clicked() { self.selected = Some(l.id); }
                let can = !l.locked;
                if ui.add_enabled(can && idx + 1 < n, egui::Button::new("^").small()).on_hover_text("Move up (drawn later = on top)").clicked() { swap = Some((idx, idx + 1)); }
                if ui.add_enabled(can && idx > 0, egui::Button::new("v").small()).on_hover_text("Move down").clicked() { swap = Some((idx, idx - 1)); }
            });
        }
        if let Some((a, b)) = swap { doc.layers.swap(a, b); }

        // ── properties of the selected layer ──
        ui.separator();
        let Some(sel_id) = self.selected else { ui.small("Select a layer to edit it."); return; };
        let Some(idx) = doc.layers.iter().position(|l| l.id == sel_id) else { self.selected = None; return; };
        let is_vis = matches!(doc.layers[idx].kind, LayerKind::Visualizer);
        let locked = doc.layers[idx].locked;
        let mut dup = false;
        let mut remove = false;

        if !self.fonts_requested && matches!(doc.layers[idx].kind, LayerKind::Text(_)) { self.request_fonts(); }

        ui.add_enabled_ui(!locked, |ui| {
            let l = &mut doc.layers[idx];
            ui.horizontal(|ui| { ui.label("Name"); ui.text_edit_singleline(&mut l.name); });
            ui.horizontal(|ui| { ui.label("Opacity"); ui.add(egui::Slider::new(&mut l.opacity, 0.0..=1.0)); });

            egui::CollapsingHeader::new("Timing and fades").default_open(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Start");
                    ui.add(egui::DragValue::new(&mut l.start_s).clamp_range(0.0..=36_000.0).speed(0.1).suffix(" s"));
                    if ui.small_button("playhead").clicked() { l.start_s = playhead; }
                });
                ui.horizontal(|ui| {
                    let mut has_end = l.end_s.is_some();
                    if ui.checkbox(&mut has_end, "End").changed() { l.end_s = if has_end { Some(playhead.max(l.start_s + 1.0)) } else { None }; }
                    if let Some(e) = &mut l.end_s {
                        ui.add(egui::DragValue::new(e).clamp_range(0.0..=36_000.0).speed(0.1).suffix(" s"));
                        if ui.small_button("playhead").clicked() { *e = playhead; }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Fade in");
                    ui.add(egui::DragValue::new(&mut l.fade_in_s).clamp_range(0.0..=30.0).speed(0.05).suffix(" s"));
                    ui.label("out");
                    ui.add(egui::DragValue::new(&mut l.fade_out_s).clamp_range(0.0..=30.0).speed(0.05).suffix(" s"));
                });
                ui.small("Tip: drag the layer's bar on the timeline, or press [ and ] to set its start / end at the playhead.");
            });

            let id = l.id;
            match &mut l.kind {
                LayerKind::Visualizer => { ui.small("Look and colours: Settings window. Layers above this one draw on top of the bars."); }
                LayerKind::Image { media, fit, offset, scale } => {
                    let w = ui.available_width();
                    ui_util::clipped_label(ui, &media.path.display().to_string(), w, true, None);
                    if ui.button("Replace file...").clicked() { self.start_pick(Pick::Replace(id)); }
                    fit_ui(ui, fit, offset, scale);
                }
                LayerKind::Video { media, fit, offset, scale, looping, mute, trim_start_s } => {
                    let w = ui.available_width();
                    ui_util::clipped_label(ui, &media.path.display().to_string(), w, true, None);
                    if ui.button("Replace file...").clicked() { self.start_pick(Pick::Replace(id)); }
                    fit_ui(ui, fit, offset, scale);
                    ui.checkbox(looping, "Loop");
                    ui.checkbox(mute, "Mute");
                    ui.horizontal(|ui| { ui.label("Start inside clip"); ui.add(egui::DragValue::new(trim_start_s).clamp_range(0.0..=36_000.0).speed(0.1).suffix(" s")); });
                    ui.small("Video sound is mixed into the export only when un-muted and not looping.");
                }
                LayerKind::Text(t) => {
                    ui.add(egui::TextEdit::multiline(&mut t.text).desired_rows(2).desired_width(f32::INFINITY));

                    // font: installed system fonts + custom file
                    ui.label("Font");
                    let current = match t.font_path.as_deref() {
                        None => "Default (built-in)".to_string(),
                        Some(p) => self.sys_fonts.iter().find(|f| f.path.to_string_lossy() == p).map(|f| f.name.clone())
                            .unwrap_or_else(|| Path::new(p).file_name().and_then(|n| n.to_str()).unwrap_or(p).to_string()),
                    };
                    egui::ComboBox::from_id_source("text_font").width((ui.available_width() - 8.0).max(120.0)).selected_text(current).show_ui(ui, |ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.font_filter).hint_text("search installed fonts"));
                        if ui.selectable_label(t.font_path.is_none(), "Default (built-in)").clicked() { t.font_path = None; }
                        if self.sys_fonts.is_empty() { ui.small(if self.fonts_rx.is_some() { "loading font list..." } else { "no fonts found" }); }
                        let needle = self.font_filter.to_lowercase();
                        egui::ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
                            for f in self.sys_fonts.iter().filter(|f| needle.is_empty() || f.name.to_lowercase().contains(&needle)) {
                                let p = f.path.to_string_lossy().into_owned();
                                if ui.selectable_label(t.font_path.as_deref() == Some(p.as_str()), &f.name).clicked() { t.font_path = Some(p); }
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Browse for a font file...").clicked() { self.start_pick(Pick::Font(id)); }
                        if t.font_path.is_some() && ui.small_button("Reset").clicked() { t.font_path = None; }
                    });

                    ui.horizontal(|ui| { ui.label("Size"); ui.add(egui::Slider::new(&mut t.size, 0.01..=0.4).text("of height")); });
                    ui.horizontal(|ui| {
                        ui.label("Colour"); ui.color_edit_button_srgb(&mut t.color);
                        for (a, n) in [(TextAlign::Left, "Left"), (TextAlign::Center, "Center"), (TextAlign::Right, "Right")] { ui.selectable_value(&mut t.align, a, n); }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Position");
                        ui.add(egui::DragValue::new(&mut t.pos[0]).clamp_range(0.0..=1.0).speed(0.005).prefix("x "));
                        ui.add(egui::DragValue::new(&mut t.pos[1]).clamp_range(0.0..=1.0).speed(0.005).prefix("y "));
                    });
                }
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.add_enabled(!is_vis, egui::Button::new("Duplicate")).clicked() { dup = true; }
            if ui.add_enabled(!is_vis && !locked, egui::Button::new("Delete")).on_disabled_hover_text("The visualizer layer cannot be deleted; hide it instead.").clicked() { remove = true; }
        });
        if dup {
            let mut c = doc.layers[idx].clone();
            c.id = doc.alloc_id();
            c.name = format!("{} copy", c.name);
            self.selected = Some(c.id);
            doc.layers.insert(idx + 1, c);
        }
        if remove { doc.layers.remove(idx); self.selected = None; }
    }
}

fn fit_ui(ui: &mut egui::Ui, fit: &mut FitMode, offset: &mut [f32; 2], scale: &mut f32) {
    ui.horizontal(|ui| {
        ui.label("Fit");
        for (f, n) in [(FitMode::Cover, "Cover"), (FitMode::Contain, "Contain"), (FitMode::Stretch, "Stretch")] { ui.selectable_value(fit, f, n); }
    });
    ui.horizontal(|ui| {
        ui.label("Scale");
        ui.add(egui::Slider::new(scale, 0.05..=5.0).logarithmic(true).suffix("x"));
        if ui.small_button("1x").clicked() { *scale = 1.0; }
    });
    ui.horizontal(|ui| {
        ui.label("Offset");
        ui.add(egui::DragValue::new(&mut offset[0]).clamp_range(-2.0..=2.0).speed(0.005).prefix("x "));
        ui.add(egui::DragValue::new(&mut offset[1]).clamp_range(-2.0..=2.0).speed(0.005).prefix("y "));
        if ui.small_button("Center").clicked() { *offset = [0.0, 0.0]; }
    });
}
