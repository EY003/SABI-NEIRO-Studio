//! Timeline: ruler, render-range strip, zoomable waveform, multi-track segments and layer bars.
//!
//! Mouse:  wheel = zoom at the pointer, Shift+wheel / middle-drag = pan, click ruler or waveform = seek.
//!         Range strip: drag empty space = new range, drag the ends = resize, drag the middle = move.
//!         Layer bars: drag = move, drag an edge = trim, double-click = whole timeline.
//!         Edges snap to the playhead, range ends, other layers and track boundaries (hold Shift for free movement).
//! The widget fills the height it is given, so resizing the bottom panel resizes the waveform.

use eframe::egui;
use visualizer_core::project::{Layer, LayerKind};
use visualizer_core::waveform::Waveform;

use crate::textshape;

const RULER_H: f32 = 18.0;
const RANGE_H: f32 = 14.0;
const ROW_H: f32 = 20.0;
const MIN_WAVE_H: f32 = 36.0;
const MAX_ROWS: usize = 8;
const HINT_H: f32 = 16.0;
const SNAP_PX: f32 = 8.0;
const GRAB_PX: f32 = 7.0;

const ACCENT: egui::Color32 = egui::Color32::from_rgb(150, 100, 255);
const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 160, 40);

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    None, Seek, Pan, RangeIn, RangeOut,
    RangeMove { grab: f64 }, RangeNew { anchor: f64 },
    LayerMove { id: u64, grab: f64 }, LayerStart(u64), LayerEnd(u64),
}

/// One audio file on the timeline (only more than one in sequence mode).
pub struct Seg { pub start: f32, pub len: f32, pub name: String, pub current: bool }

pub struct In<'a> {
    pub dur: f32,
    pub pos: f32,
    pub playing: bool,
    /// (global start of the loaded track, its waveform)
    pub wave: Option<(f32, &'a Waveform)>,
    pub segs: &'a [Seg],
    pub range: &'a mut Option<(f64, f64)>,
    pub layers: &'a mut [Layer],
    pub selected: &'a mut Option<u64>,
    pub analysing: Option<f32>,
}
pub struct Out { pub seek: Option<f32> }

pub struct Timeline { t0: f32, len: f32, drag: Drag }

struct Geo { rect: egui::Rect, ruler: egui::Rect, rng: egui::Rect, wave: egui::Rect, rows_y: f32, t0: f32, len: f32 }
impl Geo {
    fn x(&self, t: f32) -> f32 { self.rect.left() + (t - self.t0) / self.len * self.rect.width() }
    fn t(&self, x: f32) -> f32 { self.t0 + (x - self.rect.left()) / self.rect.width() * self.len }
}

fn fmt_t(t: f32) -> String {
    let t = t.max(0.0);
    if t >= 3600.0 { format!("{}:{:02}:{:02}", (t / 3600.0) as u32, ((t % 3600.0) / 60.0) as u32, (t % 60.0) as u32) }
    else { format!("{}:{:02}.{}", (t / 60.0) as u32, (t % 60.0) as u32, ((t.fract()) * 10.0) as u32) }
}

fn nice_step(px_per_s: f32) -> f32 {
    for s in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0] {
        if s * px_per_s >= 80.0 { return s; }
    }
    3600.0
}

fn hit_test(g: &Geo, p: egui::Pos2, range: Option<(f64, f64)>, layers: &[Layer], rows: &[usize], dur: f32) -> Drag {
    let t = g.t(p.x).clamp(0.0, dur) as f64;
    let near = |x: f32| (p.x - x).abs() <= GRAB_PX;
    if g.rng.contains(p) {
        return match range {
            Some((a, b)) => {
                if near(g.x(a as f32)) { Drag::RangeIn }
                else if near(g.x(b as f32)) { Drag::RangeOut }
                else if t > a && t < b { Drag::RangeMove { grab: t - a } }
                else { Drag::RangeNew { anchor: t } }
            }
            None => Drag::RangeNew { anchor: t },
        };
    }
    if g.wave.contains(p) || g.ruler.contains(p) {
        if let (Some((a, b)), true) = (range, g.wave.contains(p)) {
            if near(g.x(a as f32)) { return Drag::RangeIn; }
            if near(g.x(b as f32)) { return Drag::RangeOut; }
        }
        return Drag::Seek;
    }
    for (ri, &li) in rows.iter().enumerate() {
        let y = g.rows_y + ri as f32 * ROW_H;
        if p.y < y || p.y > y + ROW_H { continue; }
        let l = &layers[li];
        let (xs, xe) = (g.x(l.start_s as f32), g.x(l.end_s.unwrap_or(dur as f64) as f32));
        if p.x >= xs - 4.0 && p.x <= xe + 4.0 {
            if l.locked { return Drag::None; }
            return if near(xs) { Drag::LayerStart(l.id) } else if near(xe) { Drag::LayerEnd(l.id) }
                   else { Drag::LayerMove { id: l.id, grab: t - l.start_s } };
        }
    }
    Drag::None
}

impl Timeline {
    pub fn new() -> Self { Self { t0: 0.0, len: 0.0, drag: Drag::None } }
    pub fn fit(&mut self) { self.len = 0.0; self.t0 = 0.0; }
    /// Zoom around `center` (seconds); factor > 1 zooms in.
    pub fn zoom_step(&mut self, factor: f32, center: f32, dur: f32) {
        if self.len <= 0.0 { self.len = dur; }
        let new_len = (self.len / factor).clamp(dur.min(0.25), dur.max(0.25));
        self.t0 = center - (center - self.t0) * (new_len / self.len);
        self.len = new_len;
    }
    pub fn zoom_range(&mut self, a: f32, b: f32, dur: f32) {
        let pad = (b - a) * 0.08;
        self.t0 = (a - pad).max(0.0);
        self.len = ((b - a) + 2.0 * pad).clamp(dur.min(0.25), dur.max(0.25));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, inp: In) -> Out {
        let In { dur, pos, playing, wave, segs, range, layers, selected, analysing } = inp;
        let dur = dur.max(0.001);
        if self.len <= 0.0 || self.len > dur { self.len = dur; }
        self.len = self.len.max(dur.min(0.25));
        self.t0 = self.t0.clamp(0.0, (dur - self.len).max(0.0));

        let rows: Vec<usize> = layers.iter().enumerate().filter(|(_, l)| !matches!(l.kind, LayerKind::Visualizer)).map(|(i, _)| i).take(MAX_ROWS).collect();
        let rows_h = rows.len() as f32 * ROW_H;
        let wave_h = (ui.available_height() - HINT_H - RULER_H - RANGE_H - rows_h - 6.0).max(MIN_WAVE_H);
        let total_h = RULER_H + RANGE_H + wave_h + 4.0 + rows_h;
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), total_h), egui::Sense::click_and_drag());
        let w = rect.width().max(1.0);
        let shift = ui.input(|i| i.modifiers.shift);

        // ── zoom / pan ──
        if resp.hovered() {
            let (scroll, zoom, ptr) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
            let mut factor = zoom;
            if scroll.y != 0.0 && !shift { factor *= (scroll.y * 0.0035).exp(); }
            if (factor - 1.0).abs() > 1e-4 {
                let px = ptr.map(|p| p.x).unwrap_or(rect.center().x);
                let pivot = self.t0 + (px - rect.left()) / w * self.len;
                self.zoom_step(factor, pivot, dur);
            } else {
                let dx = scroll.x + if shift { scroll.y } else { 0.0 };
                if dx != 0.0 { self.t0 -= dx / w * self.len; }
            }
        }
        if playing && self.drag == Drag::None && (pos > self.t0 + self.len * 0.97 || pos < self.t0) { self.t0 = pos - self.len * 0.1; }
        self.t0 = self.t0.clamp(0.0, (dur - self.len).max(0.0));

        let g = Geo {
            rect,
            ruler: egui::Rect::from_min_size(rect.min, egui::vec2(w, RULER_H)),
            rng: egui::Rect::from_min_size(rect.min + egui::vec2(0.0, RULER_H), egui::vec2(w, RANGE_H)),
            wave: egui::Rect::from_min_size(rect.min + egui::vec2(0.0, RULER_H + RANGE_H), egui::vec2(w, wave_h)),
            rows_y: rect.top() + RULER_H + RANGE_H + wave_h + 4.0,
            t0: self.t0, len: self.len,
        };
        let px_per_s = w / g.len;

        // ── cursor feedback ──
        let hover = resp.hover_pos().filter(|_| self.drag == Drag::None).map(|p| hit_test(&g, p, *range, layers, &rows, dur));
        let icon = match self.drag {
            Drag::RangeMove { .. } | Drag::LayerMove { .. } => Some(egui::CursorIcon::Grabbing),
            Drag::RangeIn | Drag::RangeOut | Drag::LayerStart(_) | Drag::LayerEnd(_) => Some(egui::CursorIcon::ResizeHorizontal),
            Drag::Pan => Some(egui::CursorIcon::Grabbing),
            _ => match hover {
                Some(Drag::RangeIn) | Some(Drag::RangeOut) | Some(Drag::LayerStart(_)) | Some(Drag::LayerEnd(_)) => Some(egui::CursorIcon::ResizeHorizontal),
                Some(Drag::RangeMove { .. }) | Some(Drag::LayerMove { .. }) => Some(egui::CursorIcon::Grab),
                Some(Drag::RangeNew { .. }) => Some(egui::CursorIcon::Crosshair),
                _ => None,
            },
        };
        if let Some(i) = icon { ui.ctx().set_cursor_icon(i); }

        // ── snapping targets ──
        let snap = |t: f64, skip_layer: Option<u64>, skip_range: bool| -> f64 {
            if shift { return t; }
            let mut c: Vec<f64> = vec![0.0, dur as f64, pos as f64];
            if !skip_range { if let Some((a, b)) = *range { c.push(a); c.push(b); } }
            for l in layers.iter().filter(|l| Some(l.id) != skip_layer && !matches!(l.kind, LayerKind::Visualizer)) {
                c.push(l.start_s); c.push(l.end_s.unwrap_or(dur as f64));
            }
            for s in segs { c.push(s.start as f64); c.push((s.start + s.len) as f64); }
            let mut best = t;
            let mut best_d = SNAP_PX;
            for x in c { let d = ((x - t).abs() as f32) * px_per_s; if d <= best_d { best_d = d; best = x; } }
            best
        };

        // ── drag start ──
        let mut out = Out { seek: None };
        if resp.drag_started() {
            self.drag = Drag::None;
            if let Some(p) = resp.interact_pointer_pos() {
                if ui.input(|i| i.pointer.middle_down()) { self.drag = Drag::Pan; }
                else {
                    self.drag = hit_test(&g, p, *range, layers, &rows, dur);
                    if let Drag::LayerMove { id, .. } | Drag::LayerStart(id) | Drag::LayerEnd(id) = self.drag { *selected = Some(id); }
                }
            }
        }
        // ── dragging ──
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                let t = g.t(p.x).clamp(0.0, dur) as f64;
                match self.drag {
                    Drag::Seek => out.seek = Some(t as f32),
                    Drag::Pan => self.t0 = (self.t0 - resp.drag_delta().x / w * self.len).clamp(0.0, (dur - self.len).max(0.0)),
                    Drag::RangeIn => if let Some((_, b)) = *range { *range = Some((snap(t, None, true).min(b - 0.05).max(0.0), b)); },
                    Drag::RangeOut => if let Some((a, _)) = *range { *range = Some((a, snap(t, None, true).max(a + 0.05).min(dur as f64))); },
                    Drag::RangeNew { anchor } => {
                        let (a, b) = (anchor.min(snap(t, None, true)), anchor.max(snap(t, None, true)));
                        if b - a > 0.05 { *range = Some((a, b)); }
                    }
                    Drag::RangeMove { grab } => if let Some((a, b)) = *range {
                        let len = b - a;
                        let mut ns = (t - grab).clamp(0.0, (dur as f64 - len).max(0.0));
                        let (s1, s2) = (snap(ns, None, true), snap(ns + len, None, true) - len);
                        ns = if (s1 - ns).abs() <= (s2 - ns).abs() { s1 } else { s2 };
                        let ns = ns.clamp(0.0, (dur as f64 - len).max(0.0));
                        *range = Some((ns, ns + len));
                    },
                    Drag::LayerMove { id, grab } => {
                        let (s0, e0) = layers.iter().find(|l| l.id == id).map(|l| (l.start_s, l.end_s)).unwrap_or((0.0, None));
                        let len = e0.map(|e| e - s0);
                        let mut ns = (t - grab).max(0.0);
                        let s1 = snap(ns, Some(id), false);
                        let s2 = len.map(|l| snap(ns + l, Some(id), false) - l).unwrap_or(s1);
                        ns = if (s1 - ns).abs() <= (s2 - ns).abs() { s1 } else { s2 };
                        let ns = ns.max(0.0);
                        if let Some(l) = layers.iter_mut().find(|l| l.id == id) {
                            l.start_s = ns;
                            if let Some(len) = len { l.end_s = Some(ns + len); }
                        }
                    }
                    Drag::LayerStart(id) => if let Some(l) = layers.iter_mut().find(|l| l.id == id) {
                        let t = snap(t, Some(id), false);
                        l.start_s = t.min(l.end_s.unwrap_or(dur as f64) - 0.1).max(0.0);
                    },
                    Drag::LayerEnd(id) => {
                        let t = snap(t, Some(id), false);
                        if let Some(l) = layers.iter_mut().find(|l| l.id == id) { l.end_s = Some(t.max(l.start_s + 0.1).min(dur as f64)); }
                    }
                    Drag::None => {}
                }
            }
        } else if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                match hit_test(&g, p, *range, layers, &rows, dur) {
                    Drag::Seek => out.seek = Some(g.t(p.x).clamp(0.0, dur)),
                    Drag::LayerMove { id, .. } | Drag::LayerStart(id) | Drag::LayerEnd(id) => *selected = Some(id),
                    _ => {}
                }
            }
        }
        if resp.double_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                match hit_test(&g, p, *range, layers, &rows, dur) {
                    Drag::LayerMove { id, .. } | Drag::LayerStart(id) | Drag::LayerEnd(id) => {
                        if let Some(l) = layers.iter_mut().find(|l| l.id == id) { l.start_s = 0.0; l.end_s = None; }
                    }
                    Drag::RangeMove { .. } | Drag::RangeIn | Drag::RangeOut => *range = None,
                    _ => {}
                }
            }
        }
        if resp.drag_stopped() || (!resp.dragged() && !resp.drag_started()) { self.drag = Drag::None; }

        // ── drawing ──
        let (t0, len) = (self.t0, self.len);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_gray(8));
        painter.rect_filled(g.ruler, 0.0, egui::Color32::from_gray(16));
        painter.rect_filled(g.rng, 0.0, egui::Color32::from_gray(12));

        // ruler
        let step = nice_step(px_per_s);
        let mut t = (t0 / step).floor() * step;
        while t <= t0 + len {
            let x = g.x(t);
            if x >= rect.left() {
                painter.line_segment([egui::pos2(x, g.ruler.bottom() - 6.0), egui::pos2(x, g.ruler.bottom())], egui::Stroke::new(1.0_f32, egui::Color32::GRAY));
                painter.text(egui::pos2(x + 3.0, g.ruler.top() + 2.0), egui::Align2::LEFT_TOP, fmt_t(t), egui::FontId::monospace(10.0), egui::Color32::GRAY);
            }
            t += step;
        }

        // segments (several audio files)
        for (i, s) in segs.iter().enumerate() {
            let (xa, xb) = (g.x(s.start).max(rect.left()), g.x(s.start + s.len).min(rect.right()));
            if xb <= xa { continue; }
            let band = egui::Rect::from_min_max(egui::pos2(xa, g.wave.top()), egui::pos2(xb, g.wave.bottom()));
            let tint = if s.current { 26 } else if i % 2 == 0 { 14 } else { 20 };
            painter.rect_filled(band, 0.0, egui::Color32::from_gray(tint));
            if segs.len() > 1 {
                painter.line_segment([egui::pos2(xa, g.wave.top()), egui::pos2(xa, g.wave.bottom())], egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90)));
                painter.with_clip_rect(band).text(egui::pos2(xa + 4.0, g.wave.top() + 2.0), egui::Align2::LEFT_TOP,
                    textshape::shape(&s.name).as_ref(), egui::FontId::proportional(11.0), if s.current { egui::Color32::WHITE } else { egui::Color32::GRAY });
            }
        }

        // waveform of the loaded track
        match wave {
            Some((off, wf)) => {
                let a = off.max(t0);
                let b = (off + wf.duration_s).min(t0 + len);
                if b > a {
                    let (xa, xb) = (g.x(a).ceil(), g.x(b).floor());
                    let n = (xb - xa).max(1.0) as usize;
                    let (mid, h) = (g.wave.center().y, g.wave.height() * 0.46);
                    for (i, (mn, mx)) in wf.peaks(a - off, b - off, n).iter().enumerate() {
                        let x = xa + i as f32 + 0.5;
                        painter.line_segment([egui::pos2(x, mid - mx * h), egui::pos2(x, mid - mn * h)], egui::Stroke::new(1.0_f32, ACCENT));
                    }
                }
            }
            None => { painter.text(g.wave.center(), egui::Align2::CENTER_CENTER, "Add an audio file (Ctrl+I or drop it here)", egui::FontId::proportional(13.0), egui::Color32::GRAY); }
        }
        if let Some(p) = analysing {
            painter.text(g.wave.right_top() + egui::vec2(-8.0, 4.0), egui::Align2::RIGHT_TOP, format!("analysing {:.0}%", p * 100.0), egui::FontId::monospace(11.0), egui::Color32::LIGHT_YELLOW);
        }

        // render range
        if let Some((a, b)) = *range {
            let (xa, xb) = (g.x(a as f32).clamp(rect.left(), rect.right()), g.x(b as f32).clamp(rect.left(), rect.right()));
            let shade = egui::Color32::from_black_alpha(160);
            painter.rect_filled(egui::Rect::from_min_max(g.wave.left_top(), egui::pos2(xa, g.wave.bottom())), 0.0, shade);
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(xb, g.wave.top()), g.wave.right_bottom()), 0.0, shade);
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(xa, g.rng.top() + 2.0), egui::pos2(xb, g.rng.bottom() - 2.0)), 2.0, egui::Color32::from_rgba_unmultiplied(255, 160, 40, 150));
            for x in [g.x(a as f32), g.x(b as f32)] {
                if x >= rect.left() - 1.0 && x <= rect.right() + 1.0 {
                    painter.rect_filled(egui::Rect::from_center_size(egui::pos2(x, g.rng.center().y), egui::vec2(5.0, RANGE_H)), 1.0, ORANGE);
                    painter.line_segment([egui::pos2(x, g.wave.top()), egui::pos2(x, g.wave.bottom())], egui::Stroke::new(1.5_f32, ORANGE));
                }
            }
            painter.with_clip_rect(g.rng).text(egui::pos2((xa + xb) * 0.5, g.rng.center().y), egui::Align2::CENTER_CENTER,
                format!("render {}", fmt_t((b - a) as f32)), egui::FontId::monospace(10.0), egui::Color32::BLACK);
        } else {
            painter.text(g.rng.left_center() + egui::vec2(6.0, 0.0), egui::Align2::LEFT_CENTER, "drag here to choose a render range", egui::FontId::proportional(10.0), egui::Color32::from_gray(90));
        }

        // layer bars
        for (ri, &li) in rows.iter().enumerate() {
            let l = &layers[li];
            let y = g.rows_y + ri as f32 * ROW_H;
            let (s, e) = (l.start_s as f32, l.end_s.unwrap_or(dur as f64) as f32);
            let (xs, xe) = (g.x(s).max(rect.left()), g.x(e).min(rect.right()));
            if xe <= xs { continue; }
            let base = match &l.kind { LayerKind::Image { .. } => (80, 140, 220), LayerKind::Video { .. } => (160, 100, 220), LayerKind::Text(_) => (90, 190, 120), LayerKind::Visualizer => (200, 200, 200) };
            let alpha = if l.visible { 215 } else { 70 };
            let bar = egui::Rect::from_min_max(egui::pos2(xs, y + 2.0), egui::pos2(xe, y + ROW_H - 2.0));
            painter.rect_filled(bar, 3.0, egui::Color32::from_rgba_unmultiplied(base.0, base.1, base.2, alpha));
            if !l.locked {
                for x in [bar.left(), bar.right() - 3.0] { painter.rect_filled(egui::Rect::from_min_size(egui::pos2(x, bar.top()), egui::vec2(3.0, bar.height())), 1.0, egui::Color32::from_black_alpha(110)); }
            }
            if *selected == Some(l.id) { painter.rect_stroke(bar, 3.0, egui::Stroke::new(1.5_f32, egui::Color32::WHITE)); }
            let label = format!("{}  {}", l.name, fmt_t(e - s));
            painter.with_clip_rect(bar).text(bar.left_center() + egui::vec2(6.0, 0.0), egui::Align2::LEFT_CENTER, textshape::shape(&label).as_ref(), egui::FontId::proportional(11.0), egui::Color32::BLACK);
        }

        // while dragging: show what is being set
        let tip = match self.drag {
            Drag::LayerMove { id, .. } | Drag::LayerStart(id) | Drag::LayerEnd(id) => layers.iter().find(|l| l.id == id)
                .map(|l| format!("{} - {}   ({})", fmt_t(l.start_s as f32), l.end_s.map(|e| fmt_t(e as f32)).unwrap_or_else(|| "end".into()), fmt_t((l.end_s.unwrap_or(dur as f64) - l.start_s) as f32))),
            Drag::RangeIn | Drag::RangeOut | Drag::RangeNew { .. } | Drag::RangeMove { .. } => range.map(|(a, b)| format!("{} - {}   ({})", fmt_t(a as f32), fmt_t(b as f32), fmt_t((b - a) as f32))),
            _ => None,
        };
        if let (Some(tip), Some(p)) = (tip, resp.interact_pointer_pos()) {
            painter.text(p + egui::vec2(12.0, -18.0), egui::Align2::LEFT_BOTTOM, tip, egui::FontId::monospace(12.0), egui::Color32::WHITE);
        } else if let Some(h) = resp.hover_pos() {
            if g.wave.contains(h) || g.ruler.contains(h) {
                painter.line_segment([egui::pos2(h.x, g.wave.top()), egui::pos2(h.x, g.wave.bottom())], egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(70)));
                painter.text(egui::pos2(h.x + 4.0, g.wave.top() + 14.0), egui::Align2::LEFT_TOP, fmt_t(g.t(h.x)), egui::FontId::monospace(11.0), egui::Color32::WHITE);
            }
        }

        // playhead
        let px = g.x(pos);
        if px >= rect.left() && px <= rect.right() {
            painter.line_segment([egui::pos2(px, rect.top()), egui::pos2(px, g.wave.bottom() + rows_h + 4.0)], egui::Stroke::new(2.0_f32, egui::Color32::WHITE));
            painter.add(egui::Shape::convex_polygon(vec![egui::pos2(px - 5.0, rect.top()), egui::pos2(px + 5.0, rect.top()), egui::pos2(px, rect.top() + 8.0)], egui::Color32::WHITE, egui::Stroke::NONE));
        }
        ui.add(egui::Label::new(egui::RichText::new("Wheel: zoom   Shift+wheel / middle-drag: pan   Drag the strip: render range   Drag bars: move / trim   Shift: no snapping").small().weak()).wrap(false));
        out
    }
}
