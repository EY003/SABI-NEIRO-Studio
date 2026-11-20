//! Layer panel: stack order, visibility, lock, opacity, timing/fades, per-kind properties.
//! Edits `ProjectDoc::layers` in place; `ProjectState::tick` notices the change and records undo.
//! Layer 0 is the BOTTOM of the stack; the list shows the top layer first.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use eframe::egui;
use visualizer_core::project::*;

#[derive(Clone, Copy, PartialEq)]
enum Pick { Image, Video, Font(u64), Replace(u64) }

pub struct LayersUi {
    pub selected: Option<u64>,
    pick: Option<(Pick, Receiver<PathBuf>)>,
}

fn kind_label(k: &LayerKind) -> &'static str {
    match k { LayerKind::Visualizer => "Visualizer", LayerKind::Image { .. } => "Image", LayerKind::Video { .. } => "Video", LayerKind::Text(_) => "Text" }
}

impl LayersUi {
    pub fn new() -> Self { Self { selected: None, pick: None } }

    fn start_pick(&mut self, what: Pick) {
        let (tx, rx) = channel();
        self.pick = Some((what, rx));
        std::thread::spawn(move || {
            let d = rfd::FileDialog::new();
            let d = match what {
                Pick::Image => d.add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "gif"]),
                Pick::Video => d.add_filter("Video", &["mp4", "mov", "mkv", "webm", "avi", "m4v"]),
                Pick::Font(_) => d.add_filter("Fonts", &["ttf", "otf"]),
                Pick::Replace(_) => d,
            };
            if let Some(p) = d.pick_file() { let _ = tx.send(p); }
        });
    }

    fn visualizer_index(doc: &ProjectDoc) -> usize {
        doc.layers.iter().position(|l| matches!(l.kind, LayerKind::Visualizer)).unwrap_or(0)
    }

    pub fn ui(&mut self, ctx: &egui::Context, doc: &mut ProjectDoc, dir: Option<&Path>, playhead: f64) {
        // ── finished file dialog ──
        if let Some((what, rx)) = &self.pick {
            if let Ok(p) = rx.try_recv() {
                let what = *what;
                self.pick = None;
                let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("Layer").to_string();
                match what {
                    Pick::Image | Pick::Video => {
                        let id = doc.alloc_id();
                        let media = MediaRef::new(&p, dir);
                        let kind = if what == Pick::Image { LayerKind::Image { media, fit: FitMode::Cover, offset: [0.0, 0.0] } }
                                   else { LayerKind::Video { media, fit: FitMode::Cover, offset: [0.0, 0.0], looping: true, mute: true, trim_start_s: 0.0 } };
                        let at = Self::visualizer_index(doc); // backgrounds go just below the visualizer
                        doc.layers.insert(at, Layer::new(id, &name, kind));
                        self.selected = Some(id);
                    }
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
        }

        egui::SidePanel::right("layers_panel").default_width(290.0).resizable(true).show(ctx, |ui| {
            ui.heading("Layers");
            ui.horizontal_wrapped(|ui| {
                if ui.button("\u{FF0B} Image").clicked() { self.start_pick(Pick::Image); }
                if ui.button("\u{FF0B} Video").clicked() { self.start_pick(Pick::Video); }
                if ui.button("\u{FF0B} Text").clicked() {
                    let id = doc.alloc_id();
                    doc.layers.push(Layer::new(id, "Text", LayerKind::Text(TextLayer::default())));
                    self.selected = Some(id);
                }
            });
            ui.separator();

            let mut swap: Option<(usize, usize)> = None;
            let mut remove: Option<usize> = None;
            let mut dup: Option<usize> = None;
            let n_layers = doc.layers.len();
            for idx in (0..doc.layers.len()).rev() {
                let l = &mut doc.layers[idx];
                ui.horizontal(|ui| {
                    ui.checkbox(&mut l.visible, "").on_hover_text("Visible");
                    ui.toggle_value(&mut l.locked, "\u{1F512}").on_hover_text("Lock (blocks edits, reorder and delete)");
                    let sel = self.selected == Some(l.id);
                    if ui.selectable_label(sel, format!("{}  \u{00B7} {}", l.name, kind_label(&l.kind))).clicked() { self.selected = Some(l.id); }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can = !l.locked;
                        if ui.add_enabled(can && idx > 0, egui::Button::new("\u{25BC}").small()).on_hover_text("Move down").clicked() { swap = Some((idx, idx - 1)); }
                        if ui.add_enabled(can && idx + 1 < n_layers, egui::Button::new("\u{25B2}").small()).on_hover_text("Move up").clicked() { swap = Some((idx, idx + 1)); }
                    });
                });
            }
            if let Some((a, b)) = swap { if b < doc.layers.len() { doc.layers.swap(a, b); } }

            ui.separator();
            let Some(sel_id) = self.selected else { ui.small("Select a layer to edit it."); return; };
            let Some(idx) = doc.layers.iter().position(|l| l.id == sel_id) else { self.selected = None; return; };
            let l = &mut doc.layers[idx];
            let is_vis = matches!(l.kind, LayerKind::Visualizer);

            ui.add_enabled_ui(!l.locked, |ui| {
                ui.horizontal(|ui| { ui.label("Name"); ui.text_edit_singleline(&mut l.name); });
                ui.horizontal(|ui| { ui.label("Opacity"); ui.add(egui::Slider::new(&mut l.opacity, 0.0..=1.0)); });

                egui::CollapsingHeader::new("Timing & fades").default_open(false).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Start");
                        ui.add(egui::DragValue::new(&mut l.start_s).clamp_range(0.0..=36_000.0).speed(0.1).suffix(" s"));
                        if ui.small_button("playhead").clicked() { l.start_s = playhead; }
                    });
                    ui.horizontal(|ui| {
                        let mut has_end = l.end_s.is_some();
                        if ui.checkbox(&mut has_end, "End").changed() { l.end_s = if has_end { Some((playhead).max(l.start_s + 1.0)) } else { None }; }
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
                });

                let id = l.id;
                match &mut l.kind {
                    LayerKind::Visualizer => { ui.small("Look and colours: Settings window. Layers above it draw on top of the bars."); }
                    LayerKind::Image { media, fit, offset } => {
                        ui.small(media.path.display().to_string());
                        if ui.button("Replace\u{2026}").clicked() { self.start_pick(Pick::Replace(id)); }
                        fit_ui(ui, fit, offset);
                    }
                    LayerKind::Video { media, fit, offset, looping, mute, trim_start_s } => {
                        ui.small(media.path.display().to_string());
                        if ui.button("Replace\u{2026}").clicked() { self.start_pick(Pick::Replace(id)); }
                        fit_ui(ui, fit, offset);
                        ui.checkbox(looping, "Loop");
                        ui.checkbox(mute, "Mute");
                        ui.horizontal(|ui| { ui.label("Start inside clip"); ui.add(egui::DragValue::new(trim_start_s).clamp_range(0.0..=36_000.0).speed(0.1).suffix(" s")); });
                        ui.small("Video sound is mixed into the export only when un-muted and not looping.");
                    }
                    LayerKind::Text(t) => {
                        ui.add(egui::TextEdit::multiline(&mut t.text).desired_rows(2).desired_width(f32::INFINITY));
                        ui.horizontal(|ui| {
                            ui.label("Font");
                            ui.small(t.font_path.as_deref().map(|p| Path::new(p).file_name().and_then(|n| n.to_str()).unwrap_or(p).to_string()).unwrap_or_else(|| "default".into()));
                            if ui.small_button("Browse\u{2026}").clicked() { self.start_pick(Pick::Font(id)); }
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
                let locked = doc.layers[idx].locked;
                if ui.add_enabled(!is_vis, egui::Button::new("Duplicate")).clicked() { dup = Some(idx); }
                if ui.add_enabled(!is_vis && !locked, egui::Button::new("Delete")).on_disabled_hover_text("The visualizer layer cannot be deleted; hide it instead.").clicked() { remove = Some(idx); }
            });
            if let Some(i) = dup {
                let mut c = doc.layers[i].clone();
                c.id = doc.alloc_id();
                c.name = format!("{} copy", c.name);
                self.selected = Some(c.id);
                doc.layers.insert(i + 1, c);
            }
            if let Some(i) = remove { doc.layers.remove(i); self.selected = None; }
        });
    }
}

fn fit_ui(ui: &mut egui::Ui, fit: &mut FitMode, offset: &mut [f32; 2]) {
    ui.horizontal(|ui| {
        ui.label("Fit");
        for (f, n) in [(FitMode::Cover, "Cover"), (FitMode::Contain, "Contain"), (FitMode::Stretch, "Stretch")] { ui.selectable_value(fit, f, n); }
    });
    ui.horizontal(|ui| {
        ui.label("Offset");
        ui.add(egui::DragValue::new(&mut offset[0]).clamp_range(-1.0..=1.0).speed(0.005).prefix("x "));
        ui.add(egui::DragValue::new(&mut offset[1]).clamp_range(-1.0..=1.0).speed(0.005).prefix("y "));
        if ui.small_button("Center").clicked() { *offset = [0.0, 0.0]; }
    });
}
