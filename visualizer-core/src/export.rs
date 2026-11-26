//! Export model: render settings, codec/container/alpha rules, ffmpeg argument building,
//! ffmpeg discovery + capability probing, output naming and preflight validation.
//! Everything here is pure or process-spawning only; no GUI, no GL.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// ── Enums ────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Container { Mp4, Mov, Mkv, Webm }
impl Container {
    pub const ALL: [Container; 4] = [Container::Mp4, Container::Mov, Container::Mkv, Container::Webm];
    pub fn ext(self) -> &'static str { match self { Self::Mp4 => "mp4", Self::Mov => "mov", Self::Mkv => "mkv", Self::Webm => "webm" } }
    pub fn ffmpeg_format(self) -> &'static str { match self { Self::Mp4 => "mp4", Self::Mov => "mov", Self::Mkv => "matroska", Self::Webm => "webm" } }
    pub fn label(self) -> &'static str { match self { Self::Mp4 => "MP4", Self::Mov => "MOV", Self::Mkv => "MKV", Self::Webm => "WebM" } }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum VideoCodec { H264, Hevc, Av1, Vp9, ProRes, Animation }
impl VideoCodec {
    pub const ALL: [VideoCodec; 6] = [Self::H264, Self::Hevc, Self::Av1, Self::Vp9, Self::ProRes, Self::Animation];
    pub fn label(self) -> &'static str {
        match self { Self::H264 => "H.264", Self::Hevc => "H.265 / HEVC", Self::Av1 => "AV1", Self::Vp9 => "VP9",
                     Self::ProRes => "Apple ProRes", Self::Animation => "QuickTime Animation (qtrle)" }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum AudioCodec { Aac, Mp3, Opus, Flac, Pcm }
impl AudioCodec {
    pub const ALL: [AudioCodec; 5] = [Self::Aac, Self::Mp3, Self::Opus, Self::Flac, Self::Pcm];
    pub fn label(self) -> &'static str { match self { Self::Aac => "AAC", Self::Mp3 => "MP3", Self::Opus => "Opus", Self::Flac => "FLAC", Self::Pcm => "PCM 16-bit" } }
    pub fn is_lossy(self) -> bool { matches!(self, Self::Aac | Self::Mp3 | Self::Opus) }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum HwAccel { None, Nvenc, Qsv, Amf }
impl HwAccel {
    pub const ALL: [HwAccel; 4] = [Self::None, Self::Nvenc, Self::Qsv, Self::Amf];
    pub fn label(self) -> &'static str { match self { Self::None => "Software (CPU)", Self::Nvenc => "NVIDIA NVENC", Self::Qsv => "Intel Quick Sync", Self::Amf => "AMD AMF" } }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum RateControl { Quality(u8), Bitrate(u32) } // quality 1..=100, bitrate in kbps

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Collision { AutoNumber, Overwrite }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlphaSupport { Yes, Limited, No }

// ── Rules ────────────────────────────────────────────────────

pub fn container_video_ok(c: Container, v: VideoCodec) -> bool {
    use Container::*; use VideoCodec::*;
    matches!((c, v), (Mp4, H264 | Hevc | Av1) | (Mov, H264 | Hevc | ProRes | Animation)
        | (Mkv, H264 | Hevc | Av1 | Vp9 | ProRes) | (Webm, Vp9 | Av1))
}
pub fn container_audio_ok(c: Container, a: AudioCodec) -> bool {
    use AudioCodec::*; use Container::*;
    matches!((c, a), (Mp4, Aac | Mp3) | (Mov, Aac | Pcm) | (Mkv, _) | (Webm, Opus))
}
pub fn default_audio_for(c: Container) -> AudioCodec {
    match c { Container::Webm => AudioCodec::Opus, Container::Mkv => AudioCodec::Flac, _ => AudioCodec::Aac }
}
pub fn alpha_support(v: VideoCodec) -> AlphaSupport {
    match v {
        VideoCodec::Vp9 | VideoCodec::ProRes | VideoCodec::Animation => AlphaSupport::Yes,
        VideoCodec::Av1 => AlphaSupport::Limited, // only some software encoders; not offered here
        VideoCodec::H264 | VideoCodec::Hevc => AlphaSupport::No,
    }
}
pub fn hw_supported(v: VideoCodec, hw: HwAccel) -> bool {
    match (v, hw) {
        (_, HwAccel::None) => true,
        (VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Av1, _) => true,
        (VideoCodec::Vp9, HwAccel::Qsv) => true,
        _ => false,
    }
}
pub fn encoder_name(v: VideoCodec, hw: HwAccel) -> &'static str {
    let hw = if hw_supported(v, hw) { hw } else { HwAccel::None };
    match (v, hw) {
        (VideoCodec::H264, HwAccel::None) => "libx264", (VideoCodec::H264, HwAccel::Nvenc) => "h264_nvenc",
        (VideoCodec::H264, HwAccel::Qsv) => "h264_qsv", (VideoCodec::H264, HwAccel::Amf) => "h264_amf",
        (VideoCodec::Hevc, HwAccel::None) => "libx265", (VideoCodec::Hevc, HwAccel::Nvenc) => "hevc_nvenc",
        (VideoCodec::Hevc, HwAccel::Qsv) => "hevc_qsv", (VideoCodec::Hevc, HwAccel::Amf) => "hevc_amf",
        (VideoCodec::Av1, HwAccel::None) => "libsvtav1", (VideoCodec::Av1, HwAccel::Nvenc) => "av1_nvenc",
        (VideoCodec::Av1, HwAccel::Qsv) => "av1_qsv", (VideoCodec::Av1, HwAccel::Amf) => "av1_amf",
        (VideoCodec::Vp9, HwAccel::Qsv) => "vp9_qsv", (VideoCodec::Vp9, _) => "libvpx-vp9",
        (VideoCodec::ProRes, _) => "prores_ks", (VideoCodec::Animation, _) => "qtrle",
    }
}
pub fn is_hw_encoder(name: &str) -> bool { name.ends_with("_nvenc") || name.ends_with("_qsv") || name.ends_with("_amf") }

/// Quality 1..=100 -> CRF/CQ for the codec (higher quality = lower CRF).
pub fn crf_for(v: VideoCodec, quality: u8) -> u32 {
    let (worst, best) = match v { VideoCodec::H264 => (32.0, 14.0), VideoCodec::Hevc => (35.0, 16.0),
                                  VideoCodec::Av1 => (45.0, 20.0), VideoCodec::Vp9 => (45.0, 15.0), _ => (30.0, 18.0) };
    let q = (quality.clamp(1, 100) as f32 - 1.0) / 99.0;
    (worst + (best - worst) * q).round() as u32
}

/// Square-ish design scale shared by preview and export: the short side maps to 1080 logical
/// points, so every pixel-valued setting looks identical at any output resolution.
pub fn design_scale(w: u32, h: u32) -> f32 { w.min(h).max(1) as f32 / 1080.0 }

// ── Preview canvas (independent of the export resolution) ─────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum PreviewAspect { MatchExport, FitWindow, Ratio(u32, u32) }

impl PreviewAspect {
    pub const PRESETS: [(&'static str, PreviewAspect); 6] = [
        ("16:9", PreviewAspect::Ratio(16, 9)), ("9:16", PreviewAspect::Ratio(9, 16)), ("1:1", PreviewAspect::Ratio(1, 1)),
        ("4:3", PreviewAspect::Ratio(4, 3)), ("21:9", PreviewAspect::Ratio(21, 9)), ("3:4", PreviewAspect::Ratio(3, 4)),
    ];
    pub fn label(self) -> String {
        match self {
            PreviewAspect::MatchExport => "Same as export".into(),
            PreviewAspect::FitWindow => "Fill the window".into(),
            PreviewAspect::Ratio(w, h) => format!("{w}:{h}"),
        }
    }
    /// Canvas size (pixels, only the proportions and the 1080-short-side scale matter) for this mode.
    pub fn canvas(self, export: (u32, u32), window: (f32, f32)) -> (f32, f32) {
        match self {
            PreviewAspect::MatchExport => (export.0 as f32, export.1 as f32),
            PreviewAspect::FitWindow => (window.0.max(2.0), window.1.max(2.0)),
            PreviewAspect::Ratio(w, h) => {
                let (w, h) = (w.max(1) as f32, h.max(1) as f32);
                if w >= h { (1080.0 * w / h, 1080.0) } else { (1080.0, 1080.0 * h / w) }
            }
        }
    }
}

// ── Settings ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub container: Container,
    pub video_codec: VideoCodec,
    pub audio_codec: AudioCodec,
    pub hw: HwAccel,
    pub rate: RateControl,
    pub audio_bitrate_kbps: u32,
    pub transparent: bool,
    pub background: [u8; 3],
    /// Let ffmpeg use the GPU to DECODE background videos (falls back to the CPU automatically).
    pub hw_decode: bool,
    /// Keep full colour resolution (H.264 / H.265 / VP9 without alpha). Much crisper thin coloured edges,
    /// but many phones, TVs and browsers cannot play 4:4:4 video.
    pub chroma444: bool,
    /// Render only this span (seconds). `None` = whole track.
    pub range: Option<(f64, f64)>,
    pub output_dir: PathBuf,
    /// Tokens: {title} {artist} {n}
    pub name_template: String,
    pub collision: Collision,
    pub ffmpeg_path: Option<PathBuf>,
    /// Shape of the PREVIEW canvas; does not change the export unless set to `MatchExport`.
    pub preview: PreviewAspect,
    /// Play and render every audio file back to back as ONE video.
    pub sequence: bool,
}
impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 1920, height: 1080, fps: 60.0, container: Container::Mp4, video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac, hw: HwAccel::None, rate: RateControl::Quality(85),
            audio_bitrate_kbps: 192, transparent: false, background: [0, 0, 0], hw_decode: true, chroma444: false, range: None,
            output_dir: dirs::video_dir().unwrap_or_else(|| PathBuf::from(".")),
            name_template: "{artist} - {title}".into(), collision: Collision::AutoNumber, ffmpeg_path: None,
            preview: PreviewAspect::MatchExport, sequence: false,
        }
    }
}

pub fn fps_arg(fps: f32) -> String {
    for (v, s) in [(23.976, "24000/1001"), (29.97, "30000/1001"), (59.94, "60000/1001")] {
        if (fps - v).abs() < 0.01 { return s.into(); }
    }
    if (fps - fps.round()).abs() < 1e-3 { format!("{}", fps.round() as u32) } else { format!("{fps:.3}") }
}

// ── ffmpeg arguments ─────────────────────────────────────────

/// Extra audio mixed under the main track (an un-muted background video's sound).
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraAudio {
    pub path: PathBuf,
    /// Seek into the media before mixing (trim + any part of the layer already elapsed at range start).
    pub seek_s: f64,
    /// Delay before the sound starts (layer starts after the range start).
    pub delay_s: f64,
    pub volume: f32,
}

fn pix_fmt(s: &RenderSettings) -> &'static str {
    match s.video_codec {
        VideoCodec::Vp9 if s.transparent => "yuva420p",
        VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9 if s.chroma444 => "yuv444p",
        VideoCodec::ProRes => if s.transparent { "yuva444p10le" } else { "yuv422p10le" },
        VideoCodec::Animation => if s.transparent { "argb" } else { "rgb24" },
        _ => "yuv420p",
    }
}

pub fn video_args(s: &RenderSettings) -> Vec<String> {
    let hw = if hw_supported(s.video_codec, s.hw) { s.hw } else { HwAccel::None };
    let enc = encoder_name(s.video_codec, hw);
    let mut a: Vec<String> = vec!["-c:v".into(), enc.into()];
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|x| x.to_string()));
    let q = match s.rate { RateControl::Quality(q) => crf_for(s.video_codec, q).to_string(), _ => String::new() };
    let br = match s.rate { RateControl::Bitrate(k) => Some(format!("{k}k")), _ => None };
    let quality_mode = br.is_none();
    match (s.video_codec, hw) {
        (VideoCodec::ProRes, _) => push(&["-profile:v", if s.transparent { "4" } else { "3" }]),
        (VideoCodec::Animation, _) => {}
        (VideoCodec::Vp9, HwAccel::None) => {
            push(&["-row-mt", "1", "-deadline", "good", "-cpu-used", "2", "-auto-alt-ref", "0"]);
            if quality_mode { push(&["-crf", &q, "-b:v", "0"]); } else { push(&["-b:v", br.as_deref().unwrap()]); }
        }
        (_, HwAccel::None) => {
            match s.video_codec { VideoCodec::Av1 => push(&["-preset", "8"]), _ => push(&["-preset", "medium"]) }
            if quality_mode { push(&["-crf", &q]); } else { push(&["-b:v", br.as_deref().unwrap()]); }
            if s.video_codec == VideoCodec::Hevc && matches!(s.container, Container::Mp4 | Container::Mov) { push(&["-tag:v", "hvc1"]); }
        }
        (_, HwAccel::Nvenc) => {
            push(&["-preset", "p5"]);
            if quality_mode { push(&["-rc", "vbr", "-cq", &q, "-b:v", "0"]); } else { push(&["-rc", "vbr", "-b:v", br.as_deref().unwrap()]); }
            if s.video_codec == VideoCodec::Hevc && matches!(s.container, Container::Mp4 | Container::Mov) { push(&["-tag:v", "hvc1"]); }
        }
        (_, HwAccel::Qsv) => {
            if quality_mode { push(&["-global_quality", &q]); } else { push(&["-b:v", br.as_deref().unwrap()]); }
            if s.video_codec == VideoCodec::Hevc && matches!(s.container, Container::Mp4 | Container::Mov) { push(&["-tag:v", "hvc1"]); }
        }
        (_, HwAccel::Amf) => {
            push(&["-quality", "quality"]);
            if quality_mode { push(&["-rc", "cqp", "-qp_i", &q, "-qp_p", &q]); } else { push(&["-rc", "vbr_peak", "-b:v", br.as_deref().unwrap()]); }
            if s.video_codec == VideoCodec::Hevc && matches!(s.container, Container::Mp4 | Container::Mov) { push(&["-tag:v", "hvc1"]); }
        }
    }
    a.extend(["-pix_fmt".into(), pix_fmt(s).into()]);
    a
}

/// Full argument list. Frames arrive on stdin as raw RGBA, STRAIGHT (un-premultiplied) alpha,
/// bottom-up (GL order), hence `vflip`. `out` should be a temp name (`*.part`); the format is
/// forced with `-f` so the extension does not matter.
/// One audio file placed on the (global) timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSeg {
    pub path: PathBuf,
    pub start_s: f64,
    pub dur_s: f64,
    /// A silent spacer: no file, ffmpeg generates the silence.
    pub silent: bool,
}

pub fn build_args(s: &RenderSettings, audio: &[AudioSeg], out: &Path, start: f64, dur: f64, extras: &[ExtraAudio]) -> Vec<String> {
    let mut a: Vec<String> = ["-y", "-hide_banner", "-loglevel", "error", "-nostats"].iter().map(|x| x.to_string()).collect();
    a.extend(["-f", "rawvideo", "-pix_fmt", "rgba", "-video_size"].iter().map(|x| x.to_string()));
    a.push(format!("{}x{}", s.width, s.height));
    a.extend(["-framerate".into(), fps_arg(s.fps), "-i".into(), "pipe:0".into()]);

    // Audio files that overlap [start, start+dur]; one file keeps the simple fast path.
    let end = start + dur;
    let mut used: Vec<&AudioSeg> = audio.iter().filter(|g| g.start_s + g.dur_s > start && g.start_s < end).collect();
    if used.is_empty() { used = audio.iter().take(1).collect(); }
    let single = used.len() <= 1 && !used.first().map(|g| g.silent).unwrap_or(false);
    let n_audio = used.len().max(1);
    let main_label: String;
    let mut fc: Vec<String> = Vec::new();
    if single {
        let g = used[0];
        a.extend(["-ss".into(), format!("{:.6}", (start - g.start_s).max(0.0)), "-t".into(), format!("{dur:.6}"), "-i".into(), g.path.to_string_lossy().into_owned()]);
        main_label = "[1:a]".into();
    } else {
        let mut labels = String::new();
        for (i, g) in used.iter().enumerate() {
            if g.silent {
                a.extend(["-f".into(), "lavfi".into(), "-t".into(), format!("{:.6}", g.dur_s), "-i".into(), "anullsrc=channel_layout=stereo:sample_rate=48000".into()]);
            } else {
                a.extend(["-i".into(), g.path.to_string_lossy().into_owned()]);
            }
            let ls = (start - g.start_s).max(0.0);
            let le = (end - g.start_s).min(g.dur_s);
            fc.push(format!("[{}:a]atrim=start={ls:.6}:end={le:.6},asetpts=PTS-STARTPTS,aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=48000[s{i}]", i + 1));
            labels.push_str(&format!("[s{i}]"));
        }
        fc.push(format!("{labels}concat=n={}:v=0:a=1[aseq]", used.len()));
        main_label = "[aseq]".into();
    }
    for e in extras {
        a.extend(["-ss".into(), format!("{:.6}", e.seek_s), "-t".into(), format!("{dur:.6}"), "-i".into(), e.path.to_string_lossy().into_owned()]);
    }
    if extras.is_empty() {
        if single {
            a.extend(["-map", "0:v:0", "-map", "1:a:0"].iter().map(|x| x.to_string()));
        } else {
            a.extend(["-filter_complex".into(), fc.join(";"), "-map".into(), "0:v:0".into(), "-map".into(), "[aseq]".into()]);
        }
    } else {
        // amix scales every input by 1/N, so multiply back by N to keep the main track at full level.
        let n = extras.len() + 1;
        fc.push(format!("{main_label}aformat=sample_fmts=fltp:channel_layouts=stereo[m0]"));
        for (i, e) in extras.iter().enumerate() {
            let ms = (e.delay_s * 1000.0).round().max(0.0) as u64;
            fc.push(format!("[{}:a]aformat=sample_fmts=fltp:channel_layouts=stereo,adelay={ms}|{ms},volume={:.3}[m{}]", 1 + n_audio + i, e.volume, i + 1));
        }
        let mut mix = String::new();
        for i in 0..n { mix.push_str(&format!("[m{i}]")); }
        fc.push(format!("{mix}amix=inputs={n}:duration=first:dropout_transition=0,volume={n}[aout]"));
        a.extend(["-filter_complex".into(), fc.join(";"), "-map".into(), "0:v:0".into(), "-map".into(), "[aout]".into()]);
    }

    let rgb_out = s.video_codec == VideoCodec::Animation;
    let vf = if rgb_out { "vflip".to_string() } else { "vflip,scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int".to_string() };
    a.extend(["-vf".into(), vf]);
    a.extend(video_args(s));
    if !rgb_out {
        a.extend(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"].iter().map(|x| x.to_string()));
    }
    a.extend(["-r".into(), fps_arg(s.fps)]);

    match s.audio_codec {
        AudioCodec::Aac => a.extend(["-c:a".into(), "aac".into(), "-b:a".into(), format!("{}k", s.audio_bitrate_kbps)]),
        AudioCodec::Mp3 => a.extend(["-c:a".into(), "libmp3lame".into(), "-b:a".into(), format!("{}k", s.audio_bitrate_kbps)]),
        AudioCodec::Opus => a.extend(["-c:a".into(), "libopus".into(), "-b:a".into(), format!("{}k", s.audio_bitrate_kbps)]),
        AudioCodec::Flac => a.extend(["-c:a".into(), "flac".into()]),
        AudioCodec::Pcm => a.extend(["-c:a".into(), "pcm_s16le".into()]),
    }
    if matches!(s.container, Container::Mp4 | Container::Mov) { a.extend(["-movflags".into(), "+faststart".into()]); }
    a.extend(["-shortest".into(), "-f".into(), s.container.ffmpeg_format().into(), out.to_string_lossy().into_owned()]);
    a
}

// ── Naming ───────────────────────────────────────────────────

pub fn sanitize_filename(s: &str) -> String {
    let t: String = s.chars().map(|c| if "<>:\"/\\|?*".contains(c) || c.is_control() { '_' } else { c }).collect();
    let t = t.trim().trim_end_matches('.').trim().to_string();
    let reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"];
    if t.is_empty() { "untitled".into() }
    else if reserved.contains(&t.to_ascii_uppercase().as_str()) { format!("_{t}") }
    else { t }
}

pub fn output_path(s: &RenderSettings, title: &str, artist: &str, n: usize) -> PathBuf {
    let name = s.name_template.replace("{title}", title).replace("{artist}", artist).replace("{n}", &n.to_string());
    let name = if artist.is_empty() { name.replace(" - ", " ").trim().to_string() } else { name };
    s.output_dir.join(format!("{}.{}", sanitize_filename(&name), s.container.ext()))
}

/// `taken` holds paths already claimed by other queued jobs.
pub fn resolve_collision(p: PathBuf, policy: Collision, taken: &HashSet<PathBuf>) -> PathBuf {
    if policy == Collision::Overwrite && !taken.contains(&p) { return p; }
    if !p.exists() && !taken.contains(&p) { return p; }
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out").to_string();
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("mp4").to_string();
    for i in 2..10_000 {
        let c = p.with_file_name(format!("{stem} ({i}).{ext}"));
        if !c.exists() && !taken.contains(&c) { return c; }
    }
    p
}

// ── ffmpeg discovery / probing ───────────────────────────────

fn hidden(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    { use std::os::windows::process::CommandExt; cmd.creation_flags(0x0800_0000); } // CREATE_NO_WINDOW
    cmd
}
pub fn spawn_hidden(cmd: &mut Command) -> &mut Command { hidden(cmd) }

fn run_capture(path: &Path, args: &[&str]) -> Option<String> {
    let mut c = Command::new(path);
    c.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let out = hidden(&mut c).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[derive(Clone, Debug, Default)]
pub struct FfmpegCaps {
    pub path: PathBuf,
    pub version: String,
    pub encoders: HashSet<String>,
    /// Hardware encoders that actually encoded a test frame on this machine.
    pub working: HashSet<String>,
}

impl FfmpegCaps {
    pub fn discover(custom: Option<&Path>) -> Result<Self, String> {
        let mut cands: Vec<PathBuf> = vec![];
        if let Some(p) = custom { cands.push(p.to_path_buf()); }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(d) = exe.parent() { cands.push(d.join("ffmpeg.exe")); cands.push(d.join("tools").join("ffmpeg.exe")); }
        }
        cands.push(PathBuf::from("ffmpeg"));
        for p in cands {
            if let Some(v) = run_capture(&p, &["-hide_banner", "-version"]) {
                if !v.starts_with("ffmpeg version") { continue; }
                let enc = run_capture(&p, &["-hide_banner", "-encoders"]).unwrap_or_default();
                let encoders = enc.lines().filter_map(|l| {
                    let mut it = l.split_whitespace();
                    let flags = it.next()?;
                    (flags.len() == 6 && flags.starts_with('V')).then(|| it.next().map(|s| s.to_string())).flatten()
                }).collect();
                return Ok(Self { path: p, version: v.lines().next().unwrap_or("").to_string(), encoders, working: HashSet::new() });
            }
        }
        Err("ffmpeg was not found. Put ffmpeg.exe next to the app (or in a tools folder), add it to PATH, or set its path in Render settings.".into())
    }

    /// Encode a single black frame with each listed hardware encoder; keeps the ones that work.
    pub fn probe_hardware(&mut self) {
        let names: Vec<String> = self.encoders.iter().filter(|n| is_hw_encoder(n)).cloned().collect();
        for n in names {
            let mut c = Command::new(&self.path);
            c.args(["-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=1", "-frames:v", "1", "-c:v", &n, "-f", "null", "-"])
                .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
            if hidden(&mut c).status().map(|s| s.success()).unwrap_or(false) { self.working.insert(n); }
        }
    }

    pub fn encoder_usable(&self, name: &str) -> bool {
        if is_hw_encoder(name) { self.working.contains(name) } else { self.encoders.contains(name) }
    }
    pub fn hw_available(&self, v: VideoCodec, hw: HwAccel) -> bool {
        hw == HwAccel::None || (hw_supported(v, hw) && self.working.contains(encoder_name(v, hw)))
    }
}

// ── Size estimate + preflight ────────────────────────────────

pub fn estimate_bytes(s: &RenderSettings, duration_s: f64) -> u64 {
    let px = s.width as f64 * s.height as f64 * s.fps as f64;
    let vbps = match s.rate {
        RateControl::Bitrate(k) if !matches!(s.video_codec, VideoCodec::ProRes | VideoCodec::Animation) => k as f64 * 1000.0,
        RateControl::Quality(q) => {
            let k = 0.5 + q as f64 / 100.0;
            match s.video_codec {
                VideoCodec::H264 => px * 0.07 * k, VideoCodec::Hevc => px * 0.05 * k,
                VideoCodec::Av1 => px * 0.04 * k, VideoCodec::Vp9 => px * 0.05 * k,
                VideoCodec::ProRes => px * if s.transparent { 5.3 } else { 3.5 },
                VideoCodec::Animation => px * 6.0,
            }
        }
        RateControl::Bitrate(_) => px * 5.3,
    };
    let abps = if s.audio_codec.is_lossy() { s.audio_bitrate_kbps as f64 * 1000.0 } else { 1_400_000.0 };
    (((vbps + abps) / 8.0) * duration_s) as u64
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity { Error, Warning }
#[derive(Clone, Debug)]
pub struct Issue { pub severity: Severity, pub message: String }

pub struct PreflightInput<'a> {
    pub settings: &'a RenderSettings,
    pub caps: Option<&'a FfmpegCaps>,
    pub audio_path: &'a Path,
    pub audio_duration_s: Option<f32>,
    pub free_bytes: Option<u64>,
    pub out_path: &'a Path,
}

pub fn preflight(i: &PreflightInput) -> Vec<Issue> {
    let s = i.settings;
    let mut v: Vec<Issue> = vec![];
    let err = |v: &mut Vec<Issue>, m: String| v.push(Issue { severity: Severity::Error, message: m });
    let warn = |v: &mut Vec<Issue>, m: String| v.push(Issue { severity: Severity::Warning, message: m });

    match i.caps {
        None => err(&mut v, "ffmpeg not found (needed for encoding).".into()),
        Some(c) => {
            let hw = if hw_supported(s.video_codec, s.hw) { s.hw } else { HwAccel::None };
            let enc = encoder_name(s.video_codec, hw);
            if !c.encoder_usable(enc) {
                if is_hw_encoder(enc) { err(&mut v, format!("{} is not usable on this machine (no compatible GPU/driver). Choose Software.", hw.label())); }
                else { err(&mut v, format!("This ffmpeg build has no '{enc}' encoder. Use a full build (e.g. gyan.dev 'full').")); }
            }
            if s.hw != HwAccel::None && !hw_supported(s.video_codec, s.hw) {
                warn(&mut v, format!("{} cannot encode {}; software encoding will be used.", s.hw.label(), s.video_codec.label()));
            }
            if enc == "h264_nvenc" && (s.width > 4096 || s.height > 4096) { err(&mut v, "NVENC H.264 supports at most 4096 px per side. Use HEVC or software.".into()); }
        }
    }

    if !container_video_ok(s.container, s.video_codec) { err(&mut v, format!("{} cannot be stored in {}.", s.video_codec.label(), s.container.label())); }
    if !container_audio_ok(s.container, s.audio_codec) { err(&mut v, format!("{} audio cannot be stored in {}.", s.audio_codec.label(), s.container.label())); }

    if s.transparent {
        match alpha_support(s.video_codec) {
            AlphaSupport::No => err(&mut v, format!("{} has no transparency. Use WebM (VP9), MOV (ProRes 4444) or MOV (Animation), or turn transparency off.", s.video_codec.label())),
            AlphaSupport::Limited => err(&mut v, "AV1 transparency is not supported by the available encoders. Use VP9 or ProRes 4444.".into()),
            AlphaSupport::Yes => {
                if s.video_codec == VideoCodec::Vp9 { warn(&mut v, "VP9 alpha plays in browsers/OBS/Resolve but not in most desktop players or Premiere.".into()); }
                if s.video_codec == VideoCodec::ProRes { warn(&mut v, "ProRes 4444 files are very large (hundreds of Mbit/s).".into()); }
            }
        }
        if s.hw != HwAccel::None && matches!(s.video_codec, VideoCodec::Vp9) { warn(&mut v, "Hardware VP9 cannot keep alpha; transparent VP9 uses the CPU.".into()); }
    }
    if matches!(s.video_codec, VideoCodec::Vp9) && s.hw == HwAccel::Qsv && s.transparent { err(&mut v, "vp9_qsv cannot output alpha. Choose Software.".into()); }

    if s.chroma444 && !s.transparent {
        if matches!(s.video_codec, VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9) {
            warn(&mut v, "4:4:4 colour is sharper, but many phones, TVs, browsers and editors cannot play it. Turn it off for maximum compatibility.".into());
            if s.hw != HwAccel::None { warn(&mut v, "Hardware encoders often cannot encode 4:4:4; if the render fails, switch the encoder to Software.".into()); }
        } else {
            warn(&mut v, format!("{} does not support 4:4:4 here; normal 4:2:0 colour will be used.", s.video_codec.label()));
        }
    }
    if s.width < 64 || s.height < 64 || s.width > 7680 || s.height > 7680 { err(&mut v, "Resolution must be between 64 and 7680 px per side.".into()); }
    if s.width % 2 != 0 || s.height % 2 != 0 { err(&mut v, "Width and height must be even numbers for this codec.".into()); }
    if !(1.0..=240.0).contains(&s.fps) { err(&mut v, "Frame rate must be between 1 and 240.".into()); }

    match std::fs::metadata(i.audio_path) {
        Ok(m) if m.is_file() && m.len() > 0 => {}
        Ok(_) => err(&mut v, format!("Audio file is empty or not a file: {}", i.audio_path.display())),
        Err(e) => err(&mut v, format!("Audio file is missing or unreadable ({e}): {}", i.audio_path.display())),
    }
    if let (Some((a, b)), Some(d)) = (s.range, i.audio_duration_s) {
        if !(a >= 0.0 && b > a + 0.05 && b <= d as f64 + 0.05) { err(&mut v, format!("Render range {a:.1}s-{b:.1}s is outside the track (0-{d:.1}s).")); }
    }

    if let Err(e) = std::fs::create_dir_all(&s.output_dir) { err(&mut v, format!("Cannot create output folder: {e}")); }
    else {
        let probe = s.output_dir.join(".sabi_write_test");
        if std::fs::write(&probe, b"x").is_ok() { let _ = std::fs::remove_file(&probe); }
        else { err(&mut v, format!("Output folder is not writable: {}", s.output_dir.display())); }
    }
    if i.out_path.exists() {
        if s.collision == Collision::Overwrite { warn(&mut v, format!("{} exists and will be overwritten.", i.out_path.display())); }
        else { warn(&mut v, "A file with this name exists; a numbered name will be used.".into()); }
    }

    if let Some(d) = i.audio_duration_s {
        let dur = s.range.map(|(a, b)| b - a).unwrap_or(d as f64);
        let need = estimate_bytes(s, dur);
        if let Some(free) = i.free_bytes {
            if free < need { err(&mut v, format!("Not enough disk space: about {} needed, {} free.", human_bytes(need), human_bytes(free))); }
            else if free < need.saturating_mul(2) { warn(&mut v, format!("Disk space is tight: about {} needed, {} free.", human_bytes(need), human_bytes(free))); }
        }
    }
    v
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else if b >= 1e6 { format!("{:.0} MB", b / 1e6) } else { format!("{:.0} KB", b / 1e3) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_mp4_args() {
        let s = RenderSettings::default();
        let seg = AudioSeg { path: PathBuf::from("a.flac"), start_s: 0.0, dur_s: 60.0, silent: false };
        let a = build_args(&s, &[seg], Path::new("o.part"), 0.0, 10.0, &[]).join(" ");
        assert!(a.contains("-c:v libx264") && a.contains("-crf") && a.contains("yuv420p"));
        assert!(a.contains("-f mp4") && a.contains("vflip") && a.contains("-c:a aac"));
    }
    #[test]
    fn extra_audio_is_mixed() {
        let s = RenderSettings::default();
        let e = ExtraAudio { path: PathBuf::from("bg.mp4"), seek_s: 2.0, delay_s: 1.5, volume: 0.5 };
        let seg = AudioSeg { path: PathBuf::from("a.flac"), start_s: 0.0, dur_s: 60.0, silent: false };
        let a = build_args(&s, &[seg], Path::new("o.part"), 0.0, 10.0, &[e]).join(" ");
        assert!(a.contains("amix=inputs=2") && a.contains("adelay=1500|1500") && a.contains("-map [aout]"));
    }
    #[test]
    fn sequence_concat() {
        let s = RenderSettings::default();
        let segs = [AudioSeg { path: PathBuf::from("a.mp3"), start_s: 0.0, dur_s: 100.0, silent: false },
                    AudioSeg { path: PathBuf::from("b.mp3"), start_s: 100.0, dur_s: 100.0, silent: false },
                    AudioSeg { path: PathBuf::from("c.mp3"), start_s: 200.0, dur_s: 100.0, silent: false }];
        // range 90..210 touches all three; trims: a 90..100, b 0..100, c 0..10
        let a = build_args(&s, &segs, Path::new("o.part"), 90.0, 120.0, &[]).join(" ");
        assert!(a.contains("concat=n=3") && a.contains("atrim=start=90.000000:end=100.000000") && a.contains("atrim=start=0.000000:end=10.000000"));
        // a range inside one file keeps the simple path
        let a = build_args(&s, &segs, Path::new("o.part"), 110.0, 10.0, &[]).join(" ");
        assert!(!a.contains("concat") && a.contains("-ss 10.000000"));
    }
    #[test]
    fn spacer_becomes_generated_silence() {
        let s = RenderSettings::default();
        let segs = [AudioSeg { path: PathBuf::from("a.mp3"), start_s: 0.0, dur_s: 10.0, silent: false },
                    AudioSeg { path: PathBuf::new(), start_s: 10.0, dur_s: 3.0, silent: true },
                    AudioSeg { path: PathBuf::from("b.mp3"), start_s: 13.0, dur_s: 10.0, silent: false }];
        let a = build_args(&s, &segs, Path::new("o.part"), 0.0, 23.0, &[]).join(" ");
        assert!(a.contains("anullsrc") && a.contains("concat=n=3"));
    }
    #[test]
    fn preview_canvas() {
        assert_eq!(PreviewAspect::Ratio(9, 16).canvas((1920, 1080), (800.0, 600.0)), (1080.0, 1920.0));
        assert_eq!(PreviewAspect::MatchExport.canvas((1920, 1080), (800.0, 600.0)), (1920.0, 1080.0));
        assert_eq!(PreviewAspect::FitWindow.canvas((1920, 1080), (800.0, 600.0)), (800.0, 600.0));
    }
    #[test]
    fn alpha_rules() {
        let mut s = RenderSettings::default();
        s.transparent = true;
        let i = PreflightInput { settings: &s, caps: None, audio_path: Path::new("x"), audio_duration_s: None, free_bytes: None, out_path: Path::new("o.mp4") };
        assert!(preflight(&i).iter().any(|x| x.severity == Severity::Error && x.message.contains("transparency")));
        s.container = Container::Webm; s.video_codec = VideoCodec::Vp9; s.audio_codec = AudioCodec::Opus;
        assert!(video_args(&s).join(" ").contains("yuva420p"));
    }
    #[test]
    fn naming_and_crf() {
        assert_eq!(sanitize_filename("a:b/c?"), "a_b_c_");
        assert!(crf_for(VideoCodec::H264, 100) < crf_for(VideoCodec::H264, 1));
        assert_eq!(fps_arg(29.97), "30000/1001");
        assert_eq!(fps_arg(60.0), "60");
    }
    #[test]
    fn collision_numbering() {
        let mut taken = HashSet::new();
        taken.insert(PathBuf::from("zz_nonexistent/a.mp4"));
        let p = resolve_collision(PathBuf::from("zz_nonexistent/a.mp4"), Collision::AutoNumber, &taken);
        assert!(p.to_string_lossy().contains("a (2).mp4"));
    }
}
