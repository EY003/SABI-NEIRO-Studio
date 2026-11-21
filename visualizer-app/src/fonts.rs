//! System fonts: fallbacks for CJK / Arabic / Hebrew / Indic / Thai glyphs (egui's built-in fonts
//! only cover Latin, so other scripts show as squares) and a list of installed fonts for the picker.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;

use eframe::egui;

fn fonts_dir() -> PathBuf {
    let root = std::env::var("WINDIR").or_else(|_| std::env::var("SystemRoot")).unwrap_or_else(|_| "C:\\Windows".into());
    PathBuf::from(root).join("Fonts")
}

static FALLBACKS: OnceLock<Vec<(String, &'static [u8])>> = OnceLock::new();

/// Read once; the bytes are leaked on purpose so egui can borrow them without copies.
fn fallbacks() -> &'static [(String, &'static [u8])] {
    FALLBACKS.get_or_init(|| {
        let dir = fonts_dir();
        let mut v = Vec::new();
        // Order = priority for a glyph that earlier fonts lack:
        // Segoe UI (Latin ext, Greek, Cyrillic, Arabic, Hebrew) -> YaHei (Chinese + kana) -> Malgun (Korean)
        // -> Nirmala (Indic) -> Leelawadee (Thai) -> Segoe UI Symbol.
        for f in ["segoeui.ttf", "msyh.ttc", "malgun.ttf", "nirmala.ttf", "LeelawUI.ttf", "seguisym.ttf"] {
            if let Ok(bytes) = std::fs::read(dir.join(f)) {
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                v.push((format!("sys_{f}"), leaked));
            }
        }
        v
    })
}

/// Append the system fallbacks to egui's Proportional and Monospace families.
pub fn install_fallbacks(fonts: &mut egui::FontDefinitions) {
    for (name, bytes) in fallbacks() {
        fonts.font_data.insert(name.clone(), egui::FontData::from_static(*bytes));
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            if let Some(list) = fonts.families.get_mut(&fam) {
                if !list.contains(name) { list.push(name.clone()); }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SysFont { pub name: String, pub path: PathBuf }

/// Installed fonts, from the Windows font registry keys (machine-wide and per-user).
/// Runs `reg query`, so call it from a background thread.
pub fn list_system_fonts() -> Vec<SysFont> {
    let dir = fonts_dir();
    let mut out: Vec<SysFont> = Vec::new();
    for key in [r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts", r"HKCU\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"] {
        let mut c = std::process::Command::new("reg");
        c.args(["query", key]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let Ok(o) = visualizer_core::export::spawn_hidden(&mut c).output() else { continue };
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            let Some((l, r)) = line.split_once("REG_SZ") else { continue };
            let (name, file) = (l.trim(), r.trim());
            let lower = file.to_ascii_lowercase();
            if !(lower.ends_with(".ttf") || lower.ends_with(".otf") || lower.ends_with(".ttc") || lower.ends_with(".otc")) { continue; }
            let mut name = name.to_string();
            for suf in [" (TrueType)", " (OpenType)", " (All res)"] {
                if let Some(p) = name.strip_suffix(suf) { name = p.to_string(); }
            }
            if let Some((first, _)) = name.split_once(" & ") { name = first.to_string(); } // collections: first family
            let path = if file.contains(':') || file.starts_with('\\') { PathBuf::from(file) } else { dir.join(file) };
            if path.is_file() { out.push(SysFont { name, path }); }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out.dedup_by(|a, b| a.name == b.name);
    out
}
