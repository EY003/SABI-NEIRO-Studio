//! Small widgets that can never make their container wider than it is.
//! (egui panels grow to fit unbreakable text such as long CJK titles; these rows clip instead.)

use eframe::egui;

use crate::textshape;

pub fn row_height(ui: &egui::Ui) -> f32 { ui.text_style_height(&egui::TextStyle::Body) + 6.0 }

/// A selectable row of exactly `width` points; the text is shaped (Arabic/RTL) and clipped.
pub fn clipped_selectable(ui: &mut egui::Ui, selected: bool, text: &str, width: f32) -> egui::Response {
    let h = row_height(ui);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width.max(8.0), h), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let v = ui.visuals();
        let bg = if selected { v.selection.bg_fill }
                 else if resp.hovered() { v.widgets.hovered.weak_bg_fill }
                 else { egui::Color32::TRANSPARENT };
        let col = if selected { v.selection.stroke.color } else { v.text_color() };
        ui.painter().rect_filled(rect, 3.0, bg);
        let font = egui::TextStyle::Body.resolve(ui.style());
        let shaped = textshape::shape(text);
        ui.painter().with_clip_rect(rect).text(rect.left_center() + egui::vec2(6.0, 0.0), egui::Align2::LEFT_CENTER, shaped.as_ref(), font, col);
    }
    resp.on_hover_text(text)
}

/// Plain text clipped to `width` (no interaction).
pub fn clipped_label(ui: &mut egui::Ui, text: &str, width: f32, small: bool, color: Option<egui::Color32>) {
    let style = if small { egui::TextStyle::Small } else { egui::TextStyle::Body };
    let font = style.resolve(ui.style());
    let h = ui.text_style_height(&style) + 2.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width.max(8.0), h), egui::Sense::hover());
    let col = color.unwrap_or_else(|| ui.visuals().text_color());
    let shaped = textshape::shape(text);
    ui.painter().with_clip_rect(rect).text(rect.left_center(), egui::Align2::LEFT_CENTER, shaped.as_ref(), font, col);
    resp.on_hover_text(text);
}

// ── controls that reset to their default on right-click ──────────────────
// (the original Settings window already works this way; these give the studio panels the same behaviour)

use egui::emath::Numeric;

const RESET_HINT: &str = "Right-click to reset";

pub fn slider_reset<T: Numeric>(ui: &mut egui::Ui, v: &mut T, range: std::ops::RangeInclusive<T>, label: &str, default: T) -> egui::Response {
    let r = ui.add(egui::Slider::new(v, range).text(label));
    if r.secondary_clicked() { *v = default; }
    r.on_hover_text(RESET_HINT)
}

pub fn log_slider_reset(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, default: f32) -> egui::Response {
    let r = ui.add(egui::Slider::new(v, range).logarithmic(true).suffix(suffix));
    if r.secondary_clicked() { *v = default; }
    r.on_hover_text(RESET_HINT)
}

pub fn drag_reset<T: Numeric>(ui: &mut egui::Ui, v: &mut T, range: std::ops::RangeInclusive<T>, speed: f64, prefix: &str, suffix: &str, default: T) -> egui::Response {
    let r = ui.add(egui::DragValue::new(v).clamp_range(range).speed(speed).prefix(prefix).suffix(suffix));
    if r.secondary_clicked() { *v = default; }
    r.on_hover_text(RESET_HINT)
}

pub fn checkbox_reset(ui: &mut egui::Ui, v: &mut bool, label: &str, default: bool) -> egui::Response {
    let r = ui.checkbox(v, label);
    if r.secondary_clicked() { *v = default; }
    r.on_hover_text(RESET_HINT)
}

pub fn color_reset(ui: &mut egui::Ui, v: &mut [u8; 3], default: [u8; 3]) -> egui::Response {
    let r = ui.color_edit_button_srgb(v);
    if r.secondary_clicked() { *v = default; }
    r.on_hover_text(RESET_HINT)
}
