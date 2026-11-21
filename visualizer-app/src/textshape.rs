//! Text shaping that egui lacks: Arabic/Persian letter joining (presentation forms) and
//! right-to-left reordering. Text without RTL characters is returned untouched.
//! (CJK needs no shaping, only a font with the glyphs: see fonts.rs.)

use std::borrow::Cow;

fn is_rtl_char(c: char) -> bool {
    matches!(c as u32, 0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF)
}

pub fn shape(text: &str) -> Cow<'_, str> {
    if !text.chars().any(is_rtl_char) { return Cow::Borrowed(text); }
    let reshaped = reshape_arabic(text);
    let bidi = unicode_bidi::BidiInfo::new(&reshaped, None);
    let mut out = String::with_capacity(reshaped.len());
    for para in &bidi.paragraphs {
        let line = para.range.clone();
        out.push_str(&bidi.reorder_line(para, line));
    }
    Cow::Owned(out)
}

/// [isolated, final, initial, medial]; 0 = that form does not exist for the letter.
fn forms(c: char) -> Option<[u32; 4]> {
    Some(match c as u32 {
        0x0621 => [0xFE80, 0, 0, 0],
        0x0622 => [0xFE81, 0xFE82, 0, 0],
        0x0623 => [0xFE83, 0xFE84, 0, 0],
        0x0624 => [0xFE85, 0xFE86, 0, 0],
        0x0625 => [0xFE87, 0xFE88, 0, 0],
        0x0626 => [0xFE89, 0xFE8A, 0xFE8B, 0xFE8C],
        0x0627 => [0xFE8D, 0xFE8E, 0, 0],
        0x0628 => [0xFE8F, 0xFE90, 0xFE91, 0xFE92],
        0x0629 => [0xFE93, 0xFE94, 0, 0],
        0x062A => [0xFE95, 0xFE96, 0xFE97, 0xFE98],
        0x062B => [0xFE99, 0xFE9A, 0xFE9B, 0xFE9C],
        0x062C => [0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0],
        0x062D => [0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4],
        0x062E => [0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8],
        0x062F => [0xFEA9, 0xFEAA, 0, 0],
        0x0630 => [0xFEAB, 0xFEAC, 0, 0],
        0x0631 => [0xFEAD, 0xFEAE, 0, 0],
        0x0632 => [0xFEAF, 0xFEB0, 0, 0],
        0x0633 => [0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4],
        0x0634 => [0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8],
        0x0635 => [0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC],
        0x0636 => [0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0],
        0x0637 => [0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4],
        0x0638 => [0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8],
        0x0639 => [0xFEC9, 0xFECA, 0xFECB, 0xFECC],
        0x063A => [0xFECD, 0xFECE, 0xFECF, 0xFED0],
        0x0641 => [0xFED1, 0xFED2, 0xFED3, 0xFED4],
        0x0642 => [0xFED5, 0xFED6, 0xFED7, 0xFED8],
        0x0643 => [0xFED9, 0xFEDA, 0xFEDB, 0xFEDC],
        0x0644 => [0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0],
        0x0645 => [0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4],
        0x0646 => [0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8],
        0x0647 => [0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC],
        0x0648 => [0xFEED, 0xFEEE, 0, 0],
        0x0649 => [0xFEEF, 0xFEF0, 0, 0],
        0x064A => [0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4],
        // Persian / Urdu additions
        0x067E => [0xFB56, 0xFB57, 0xFB58, 0xFB59],
        0x0686 => [0xFB7A, 0xFB7B, 0xFB7C, 0xFB7D],
        0x0698 => [0xFB8A, 0xFB8B, 0, 0],
        0x06A9 => [0xFB8E, 0xFB8F, 0xFB90, 0xFB91],
        0x06AF => [0xFB92, 0xFB93, 0xFB94, 0xFB95],
        0x06CC => [0xFBFC, 0xFBFD, 0xFBFE, 0xFBFF],
        _ => return None,
    })
}

fn is_mark(c: char) -> bool { matches!(c as u32, 0x064B..=0x065F | 0x0670) } // harakat: transparent for joining
fn joins_next(c: char) -> bool { c == '\u{0640}' || forms(c).map(|f| f[2] != 0).unwrap_or(false) }
fn joins_prev(c: char) -> bool { c == '\u{0640}' || forms(c).is_some() }

fn reshape_arabic(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let prev_of = |i: usize| -> Option<char> { (0..i).rev().map(|k| chars[k]).find(|c| !is_mark(*c)) };
    let next_of = |i: usize| -> Option<(usize, char)> { (i + 1..chars.len()).map(|k| (k, chars[k])).find(|(_, c)| !is_mark(*c)) };
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let Some(f) = forms(c) else { out.push(c); i += 1; continue };
        let prev_joins = prev_of(i).map(joins_next).unwrap_or(false);

        // lam + alef is one ligature glyph
        if c == '\u{0644}' {
            if let Some((k, n)) = next_of(i) {
                let lig = match n as u32 {
                    0x0622 => Some((0xFEF5, 0xFEF6)), 0x0623 => Some((0xFEF7, 0xFEF8)),
                    0x0625 => Some((0xFEF9, 0xFEFA)), 0x0627 => Some((0xFEFB, 0xFEFC)), _ => None,
                };
                if let Some((iso, fin)) = lig {
                    out.push(char::from_u32(if prev_joins { fin } else { iso }).unwrap_or(c));
                    for m in i + 1..k { out.push(chars[m]); }
                    i = k + 1;
                    continue;
                }
            }
        }
        let next_joins = joins_next(c) && next_of(i).map(|(_, n)| joins_prev(n)).unwrap_or(false);
        let pick = |wanted: u32, fallback: u32| if wanted != 0 { wanted } else { fallback };
        let form = match (prev_joins, next_joins) {
            (true, true) => pick(f[3], pick(f[1], f[0])),
            (true, false) => pick(f[1], f[0]),
            (false, true) => pick(f[2], f[0]),
            (false, false) => f[0],
        };
        out.push(char::from_u32(form).unwrap_or(c));
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latin_is_untouched() {
        assert!(matches!(shape("Hello \u{4e16}\u{754c}"), Cow::Borrowed(_)));
    }
    #[test]
    fn arabic_gets_joined_forms_and_is_reversed() {
        // "بسم": beh initial, seen medial, meem final -> visual (RTL) order is reversed
        let s = shape("\u{0628}\u{0633}\u{0645}");
        let v: Vec<u32> = s.chars().map(|c| c as u32).collect();
        assert_eq!(v, vec![0xFEE2, 0xFEB4, 0xFE91]);
    }
    #[test]
    fn lam_alef_ligature() {
        let s = reshape_arabic("\u{0644}\u{0627}");
        assert_eq!(s.chars().next().map(|c| c as u32), Some(0xFEFB));
    }
}
