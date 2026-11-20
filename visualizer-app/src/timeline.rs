//! Timeline: ruler, zoomable waveform, playhead scrub, render-range (in/out) markers and
//! draggable layer bars (move / trim start / trim end). Ctrl+wheel zooms around the pointer,
//! wheel or middle-drag pans; the view follows the playhead during playback.

use eframe::egui;
use visualizer_core::project::{Layer, LayerKind};
use visualizer_core::waveform::Waveform;

const RULER_H: f32 = 16.0;
const WAVE_H: f32 = 84.0;
const ROW_H: f32 = 16.0;
const MAX_ROWS: usize = 6;

#[derive(Clone, Copy, PartialEq)]
enum Drag { None, Seek, RangeIn, RangeOut, Pan, LayerMove { id: u64, grab: f64 }, LayerStart(u64), LayerEnd(u64) }

pub struct Timeline { t0: f32, len: f32, drag: Drag }

pub struct In<'a> {
    pub dur: f32,
    pub pos: f32,
    pub playing: bool,
    pub wave: Option<&'a Waveform>,
    pub range: &'a mut Option<(f64, f64)>,
    pub layers: &'a mut [Layer],
    pub selected: &'a mut Option<u64>,
    pub analysing: Option<f32>,
}
pub struct Out { pub seek: Option<f32> }

fn fmt_t(t: f32) -> String { let t = t.max(0.0); format!("{}:{:02}", (t / 60.0) as u32, (t % 60.0) as u32) }

fn nice_step(px_per_s: f32) -> f32 {
    for s in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0] {
        if s * px_per_s >= 70.0 { return s; }
    }
    1200.0
}

impl Timeline {
    pub fn new() -> Self { Self { t0: 0.0, len: 0.0, drag: Drag::None } }
    pub fn fit(&mut self) { self.len = 0.0; self.t0 = 0.0; }

    pub fn ui(&mut self, ui: &mut egui::Ui, inp: In) -> Out {
        let dur = inp.dur.max(0.001);
        if self.len <= 0.0 || self.len > dur { self.len = dur; }
        self.len = self.len.max(dur.min(0.5));
        self.t0 = self.t0.clamp(0.0, (dur - self.len).max(0.0));

        let rows: Vec<usize> = inp.layers.iter().enumerate()
            .filter(|(_, l)| !matches!(l.kind, LayerKind::Visualizer)).map(|(i, _)| i).take(MAX_ROWS).collect();
        let total_h = RULER_H + WAVE_H + 4.0 + rows.len() as f32 * ROW_H;
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), total_h), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let wave_r = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, RULER_H), egui::vec2(rect.width(), WAVE_H));
        let w = rect.width().max(1.0);

        // ── zoom / pan input ──
        if resp.hovered() {
            let (zoom, scroll, ptr) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta, i.pointer.hover_pos()));
            if (zoom - 1.0).abs() > 1e-4 {
                let px = ptr.map(|p| p.x).unwrap_or(rect.center().x);
                let pivot = self.t0 + (px - rect.left()) / w * self.len;
                let new_len = (self.len / zoom).clamp(dur.min(0.5), dur);
                self.t0 = pivot - (pivot - self.t0) * (new_len / self.len);
                self.len = new_len;
            } else if scroll != egui::Vec2::ZERO {
                self.t0 -= (scroll.x + scroll.y) / w * self.len;
            }
        }
        if inp.playing && self.drag == Drag::None && (inp.pos > self.t0 + self.len * 0.97 || inp.pos < self.t0) {
            self.t0 = inp.pos - self.len * 0.1; // page the view along with the playhead
        }
        self.t0 = self.t0.clamp(0.0, (dur - self.len).max(0.0));
        let (t0, len) = (self.t0, self.len);
        let x_of = |t: f32| rect.left() + (t - t0) / len * w;
        let t_of = |x: f32| t0 + (x - rect.left()) / w * len;

        // ── interaction ──
        let mut out = Out { seek: None };
        let middle = ui.input(|i| i.pointer.middle_down());
        if resp.drag_started() {
            self.drag = Drag::None;
            if let Some(p) = resp.interact_pointer_pos() {
                let t = t_of(p.x).clamp(0.0, dur);
                if middle { self.drag = Drag::Pan; }
                else if let Some((a, b)) = *inp.range {
                    if wave_r.contains(p) && (p.x - x_of(a as f32)).abs() <= 7.0 { self.drag = Drag::RangeIn; }
                    else if wave_r.contains(p) && (p.x - x_of(b as f32)).abs() <= 7.0 { self.drag = Drag::RangeOut; }
                }
                if self.drag == Drag::None {
                    for (ri, &li) in rows.iter().enumerate() {
                        let row = egui::Rect::from_min_size(egui::pos2(rect.left(), wave_r.bottom() + 4.0 + ri as f32 * ROW_H), egui::vec2(w, ROW_H));
                        if !row.contains(p) { continue; }
                        let l = &inp.layers[li];
                        let (s, e) = (l.start_s, l.end_s.unwrap_or(dur as f64));
                        let (xs, xe) = (x_of(s as f32), x_of(e as f32));
                        if p.x >= xs - 4.0 && p.x <= xe + 4.0 {
                            *inp.selected = Some(l.id);
                            if !l.locked {
                                self.drag = if (p.x - xs).abs() <= 5.0 { Drag::LayerStart(l.id) }
                                            else if (p.x - xe).abs() <= 5.0 { Drag::LayerEnd(l.id) }
                                            else { Drag::LayerMove { id: l.id, grab: t as f64 - s } };
                            }
                        }
                        break;
                    }
                }
                if self.drag == Drag::None && (p.y < wave_r.bottom() + 2.0) { self.drag = Drag::Seek; }
            }
        }
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                let t = t_of(p.x).clamp(0.0, dur) as f64;
                match self.drag {
                    Drag::Seek => out.seek = Some(t as f32),
                    Drag::Pan => self.t0 = (self.t0 - resp.drag_delta().x / w * self.len).clamp(0.0, (dur - self.len).max(0.0)),
                    Drag::RangeIn => if let Some((_, b)) = inp.range.as_mut() { let nb = *b; *inp.range = Some((t.min(nb - 0.05), nb)); },
                    Drag::RangeOut => if let Some((a, _)) = inp.range.as_mut() { let na = *a; *inp.range = Some((na, t.max(na + 0.05))); },
                    Drag::LayerMove { id, grab } => if let Some(l) = inp.layers.iter_mut().find(|l| l.id == id) {
                        let ns = (t - grab).max(0.0);
                        let d = ns - l.start_s;
                        l.start_s = ns;
                        if let Some(e) = l.end_s.as_mut() { *e = (*e + d).max(ns + 0.1); }
                    },
                    Drag::LayerStart(id) => if let Some(l) = inp.layers.iter_mut().find(|l| l.id == id) { l.start_s = t.min(l.end_s.unwrap_or(dur as f64) - 0.1).max(0.0); },
                    Drag::LayerEnd(id) => if let Some(l) = inp.layers.iter_mut().find(|l| l.id == id) { l.end_s = Some(t.max(l.start_s + 0.1).min(dur as f64)); },
                    Drag::None => {}
                }
            }
        } else if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() { if p.y < wave_r.bottom() + 2.0 { out.seek = Some(t_of(p.x).clamp(0.0, dur)); } }
        }
        if resp.drag_released() || (!resp.dragged() && !resp.drag_started()) { self.drag = Drag::None; }

        // ── drawing ──
        painter.rect_filled(rect, 4.0, egui::Color32::from_gray(22));
        // ruler
        let step = nice_step(w / len);
        let mut t = (t0 / step).floor() * step;
        while t <= t0 + len {
            let x = x_of(t);
            if x >= rect.left() {
                painter.line_segment([egui::pos2(x, rect.top() + RULER_H - 5.0), egui::pos2(x, rect.top() + RULER_H)], egui::Stroke::new(1.0, egui::Color32::GRAY));
                painter.text(egui::pos2(x + 3.0, rect.top() + 1.0), egui::Align2::LEFT_TOP, fmt_t(t), egui::FontId::monospace(10.0), egui::Color32::GRAY);
            }
            t += step;
        }
        // waveform
        match inp.wave {
            Some(wf) => {
                let n = rect.width().max(1.0) as usize;
                let (mid, h) = (wave_r.center().y, wave_r.height() * 0.46);
                let col = egui::Color32::from_rgb(110, 170, 255);
                for (i, (mn, mx)) in wf.peaks(t0, t0 + len, n).iter().enumerate() {
                    let x = rect.left() + i as f32 + 0.5;
                    painter.line_segment([egui::pos2(x, mid - mx * h), egui::pos2(x, mid - mn * h)], egui::Stroke::new(1.0, col));
                }
            }
            None => { painter.text(wave_r.center(), egui::Align2::CENTER_CENTER, "Add an audio file (Ctrl+I)", egui::FontId::proportional(13.0), egui::Color32::GRAY); }
        }
        if let Some(p) = inp.analysing { painter.text(wave_r.left_top() + egui::vec2(6.0, 4.0), egui::Align2::LEFT_TOP, format!("analysing {:.0}%", p * 100.0), egui::FontId::monospace(11.0), egui::Color32::LIGHT_YELLOW); }
        // render range
        if let Some((a, b)) = *inp.range {
            let shade = egui::Color32::from_black_alpha(150);
            let (xa, xb) = (x_of(a as f32).clamp(rect.left(), rect.right()), x_of(b as f32).clamp(rect.left(), rect.right()));
            painter.rect_filled(egui::Rect::from_min_max(wave_r.left_top(), egui::pos2(xa, wave_r.bottom())), 0.0, shade);
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(xb, wave_r.top()), wave_r.right_bottom()), 0.0, shade);
            let orange = egui::Color32::from_rgb(255, 160, 40);
            for (x, label) in [(x_of(a as f32), "IN"), (x_of(b as f32), "OUT")] {
                if x >= rect.left() - 1.0 && x <= rect.right() + 1.0 {
                    painter.line_segment([egui::pos2(x, wave_r.top()), egui::pos2(x, wave_r.bottom())], egui::Stroke::new(2.0, orange));
                    painter.text(egui::pos2(x + 3.0, wave_r.bottom() - 2.0), egui::Align2::LEFT_BOTTOM, label, egui::FontId::monospace(10.0), orange);
                }
            }
        }
        // layer bars
        for (ri, &li) in rows.iter().enumerate() {
            let l = &inp.layers[li];
            let y = wave_r.bottom() + 4.0 + ri as f32 * ROW_H;
            let (xs, xe) = (x_of(l.start_s as f32).max(rect.left()), x_of(l.end_s.unwrap_or(dur as f64) as f32).min(rect.right()));
            if xe <= xs { continue; }
            let base = match &l.kind { LayerKind::Image { .. } => (80, 140, 220), LayerKind::Video { .. } => (160, 100, 220), LayerKind::Text(_) => (90, 190, 120), LayerKind::Visualizer => (200, 200, 200) };
            let a = if l.visible { 200 } else { 70 };
            let bar = egui::Rect::from_min_max(egui::pos2(xs, y + 1.0), egui::pos2(xe, y + ROW_H - 1.0));
            painter.rect_filled(bar, 3.0, egui::Color32::from_rgba_unmultiplied(base.0, base.1, base.2, a));
            if *inp.selected == Some(l.id) { painter.rect_stroke(bar, 3.0, egui::Stroke::new(1.5, egui::Color32::WHITE)); }
            painter.with_clip_rect(bar).text(bar.left_center() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, &l.name, egui::FontId::proportional(10.0), egui::Color32::BLACK);
        }
        // hover + playhead
        if let Some(h) = resp.hover_pos() {
            if rect.contains(h) {
                painter.line_segment([egui::pos2(h.x, wave_r.top()), egui::pos2(h.x, wave_r.bottom())], egui::Stroke::new(1.0, egui::Color32::from_white_alpha(70)));
                painter.text(egui::pos2(h.x + 4.0, wave_r.top() + 2.0), egui::Align2::LEFT_TOP, fmt_t(t_of(h.x)), egui::FontId::monospace(11.0), egui::Color32::WHITE);
            }
        }
        let px = x_of(inp.pos);
        if px >= rect.left() && px <= rect.right() {
            painter.line_segment([egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())], egui::Stroke::new(2.0, egui::Color32::WHITE));
        }
        out
    }
}
