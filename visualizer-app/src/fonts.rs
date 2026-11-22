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

/// Scripts that need a big system font (loaded only while some visible text uses them).
pub const NEED_CJK: u8 = 1;
pub const NEED_HANGUL: u8 = 2;

/// Which big fonts does this text need?
pub fn needs_of(s: &str) -> u8 {
    let mut n = 0;
    for c in s.chars() {
        let u = c as u32;
        if u < 0x1100 { continue; } // Latin, Greek, Cyrillic, Hebrew, Arabic ... (small fonts cover these)
        match u {
            0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7AF => n |= NEED_HANGUL,
            0x2E80..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FFFF => n |= NEED_CJK,
            _ => {}
        }
    }
    n
}

/// Small fonts (about 4 MB together) are cached; the big CJK / Hangul fonts are read on demand and
/// dropped with the font definitions when no text needs them any more (nothing is leaked).
fn small_fallbacks() -> &'static [(String, Vec<u8>)] {
    static SMALL: OnceLock<Vec<(String, Vec<u8>)>> = OnceLock::new();
    SMALL.get_or_init(|| {
        let dir = fonts_dir();
        // Segoe UI (Arabic, Hebrew, Latin extended) -> Nirmala UI (Indic) -> Leelawadee UI (Thai) -> Segoe UI Symbol
        ["segoeui.ttf", "nirmala.ttf", "LeelawUI.ttf", "seguisym.ttf"].iter()
            .filter_map(|f| std::fs::read(dir.join(f)).ok().map(|b| (format!("sys_{f}"), b)))
            .collect()
    })
}

/// Append the system fallbacks to egui's Proportional and Monospace families.
pub fn install_fallbacks(fonts: &mut egui::FontDefinitions, needs: u8) {
    let mut add = |name: String, bytes: Vec<u8>| {
        fonts.font_data.insert(name.clone(), egui::FontData::from_owned(bytes));
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            if let Some(list) = fonts.families.get_mut(&fam) {
                if !list.contains(&name) { list.push(name.clone()); }
            }
        }
    };
    for (name, bytes) in small_fallbacks() { add(name.clone(), bytes.clone()); }
    let dir = fonts_dir();
    if needs & NEED_CJK != 0 {
        if let Ok(b) = std::fs::read(dir.join("msyh.ttc")) { add("sys_msyh.ttc".into(), b); }     // Chinese + kana
    }
    if needs & NEED_HANGUL != 0 {
        if let Ok(b) = std::fs::read(dir.join("malgun.ttf")) { add("sys_malgun.ttf".into(), b); } // Korean
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
