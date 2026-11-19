//! Window modes: transparent desktop Overlay <-> normal Window <-> Fullscreen.
//! Mode and the normal-window rectangle persist in window.toml.
//! NOTE: verify every `ViewportCommand` variant against egui 0.27.

use eframe::egui;
use serde::{Deserialize, Serialize};

use crate::overlay;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WindowMode { #[default] Overlay, Window, Fullscreen }

impl WindowMode {
    pub const ALL: [WindowMode; 3] = [WindowMode::Overlay, WindowMode::Window, WindowMode::Fullscreen];
    pub fn label(self) -> &'static str {
        match self { WindowMode::Overlay => "Desktop overlay", WindowMode::Window => "Window", WindowMode::Fullscreen => "Fullscreen" }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct WindowPrefs {
    pub mode: WindowMode,
    /// Outer position / inner size of the normal window, in physical pixels.
    pub pos: [f32; 2],
    pub size: [f32; 2],
}
impl Default for WindowPrefs {
    fn default() -> Self { Self { mode: WindowMode::Overlay, pos: [120.0, 80.0], size: [1280.0, 720.0] } }
}

impl WindowPrefs {
    fn path() -> std::path::PathBuf {
        dirs::config_dir().unwrap_or_else(|| ".".into()).join("RustVisualizer").join("window.toml")
    }
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path()).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }
    pub fn save(&self) {
        let p = Self::path();
        if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
        if let Ok(s) = toml::to_string(self) { let _ = std::fs::write(p, s); }
    }

    /// Monitor layout may have changed since last run (unplugged display, new scaling):
    /// if less than 120x80 px of the saved window is on any monitor, fall back to the
    /// primary monitor's top-left area.
    pub fn sanitized(mut self, monitors: &[[i32; 4]]) -> Self {
        self.size[0] = self.size[0].clamp(320.0, 16_384.0);
        self.size[1] = self.size[1].clamp(240.0, 16_384.0);
        if monitors.is_empty() { return self; }
        let (x0, y0, x1, y1) = (self.pos[0], self.pos[1], self.pos[0] + self.size[0], self.pos[1] + self.size[1]);
        let visible = monitors.iter().any(|m| {
            let w = (x1.min(m[2] as f32) - x0.max(m[0] as f32)).max(0.0);
            let h = (y1.min(m[3] as f32) - y0.max(m[1] as f32)).max(0.0);
            w >= 120.0 && h >= 80.0
        });
        if !visible {
            let m = monitors[0];
            self.pos = [m[0] as f32 + 80.0, m[1] as f32 + 80.0];
            self.size[0] = self.size[0].min((m[2] - m[0]) as f32 - 160.0);
            self.size[1] = self.size[1].min((m[3] - m[1]) as f32 - 160.0);
        }
        self
    }

    /// Call every frame while in `Window` mode; remembers where the user put the window.
    pub fn remember(&mut self, ctx: &egui::Context) {
        let (outer, inner) = ctx.input(|i| (i.viewport().outer_rect, i.viewport().inner_rect));
        if let (Some(o), Some(i)) = (outer, inner) {
            if i.width() > 100.0 && i.height() > 100.0 {
                self.pos = [o.min.x, o.min.y];
                self.size = [i.width(), i.height()];
            }
        }
    }
}

/// F11 toggles Window/Fullscreen, Esc leaves fullscreen. Key events only arrive when the
/// window is interactive, so this is a no-op in Overlay mode.
pub fn shortcut(ctx: &egui::Context, mode: WindowMode) -> Option<WindowMode> {
    if mode == WindowMode::Overlay { return None; }
    let f11 = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F11));
    if f11 {
        return Some(if mode == WindowMode::Fullscreen { WindowMode::Window } else { WindowMode::Fullscreen });
    }
    if mode == WindowMode::Fullscreen && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
        return Some(WindowMode::Window);
    }
    None
}

/// Perform the switch. The caller sets its own `mode` field and `is_passthrough_set = false`.
pub fn transition(
    ctx: &egui::Context,
    overlay_slot: &mut Option<overlay::Overlay>,
    from: WindowMode,
    to: WindowMode,
    prefs: &WindowPrefs,
) {
    use egui::ViewportCommand as C;
    match to {
        WindowMode::Overlay => {
            ctx.send_viewport_cmd(C::Fullscreen(false));
            ctx.send_viewport_cmd(C::Decorations(false));
            ctx.send_viewport_cmd(C::Resizable(false));
            // `ensure_overlay` re-attaches the hook next frame (overlay_slot is None);
            // the existing overlay path then resizes/positions the window as before.
        }
        WindowMode::Window | WindowMode::Fullscreen => {
            if from == WindowMode::Overlay {
                if let Some(ov) = overlay_slot.as_mut() { ov.detach(); }
                *overlay_slot = None;
                ctx.send_viewport_cmd(C::MousePassthrough(false));
                ctx.send_viewport_cmd(C::Decorations(true));
                ctx.send_viewport_cmd(C::Resizable(true));
                ctx.send_viewport_cmd(C::WindowLevel(egui::WindowLevel::Normal));
                ctx.send_viewport_cmd(C::InnerSize(egui::vec2(prefs.size[0], prefs.size[1])));
                ctx.send_viewport_cmd(C::OuterPosition(egui::pos2(prefs.pos[0], prefs.pos[1])));
            }
            ctx.send_viewport_cmd(C::Fullscreen(to == WindowMode::Fullscreen));
        }
    }
}
