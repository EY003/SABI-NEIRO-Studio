//! Whole-file audio decoding (symphonia). Output is interleaved f32 in memory:
//! simple seeking, waveform building, analysis and export all read the same buffer.

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision, StandardTagKey};
use symphonia::core::probe::Hint;

#[derive(Clone, Debug, Default)]
pub struct TrackMeta {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub cover: Option<Vec<u8>>,
}

pub struct DecodedAudio {
    pub samples: Vec<f32>, // interleaved
    pub channels: usize,
    pub sample_rate: u32,
    pub meta: TrackMeta,
}

impl DecodedAudio {
    pub fn duration_s(&self) -> f32 {
        self.samples.len() as f32 / self.channels.max(1) as f32 / self.sample_rate.max(1) as f32
    }
}

#[derive(Debug)]
pub enum DecodeError {
    Io(String),
    Unsupported(String),
    Cancelled,
    Empty,
}
impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Io(e) => write!(f, "Cannot read file: {e}"),
            DecodeError::Unsupported(e) => write!(f, "Unsupported or corrupt audio: {e}"),
            DecodeError::Cancelled => write!(f, "Cancelled"),
            DecodeError::Empty => write!(f, "The file contains no audio"),
        }
    }
}
impl std::error::Error for DecodeError {}

fn read_rev(rev: &MetadataRevision, m: &mut TrackMeta) {
    for tag in rev.tags() {
        match tag.std_key {
            Some(StandardTagKey::TrackTitle) if m.title.is_empty() => m.title = tag.value.to_string(),
            Some(StandardTagKey::Artist) if m.artist.is_empty() => m.artist = tag.value.to_string(),
            Some(StandardTagKey::Album) if m.album.is_empty() => m.album = tag.value.to_string(),
            _ => {}
        }
    }
    if m.cover.is_none() {
        if let Some(v) = rev.visuals().first() {
            m.cover = Some(v.data.to_vec());
        }
    }
}

pub fn decode_file(path: &Path, cancel: &AtomicBool) -> Result<DecodedAudio, DecodeError> {
    let file = File::open(path).map_err(|e| DecodeError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DecodeError::Unsupported(e.to_string()))?;

    let mut meta = TrackMeta::default();
    {
        if let Some(md) = probed.metadata.get() {
            if let Some(rev) = md.current() { read_rev(rev, &mut meta); }
        }
    }
    let mut format = probed.format;
    if let Some(rev) = format.metadata().current() { read_rev(rev, &mut meta); }
    if meta.title.is_empty() {
        meta.title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    }

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::Unsupported("no audio track".into()))?;
    let track_id = track.id;
    // Reserve exactly what the container announces: growing by doubling can briefly use 2x the memory.
    let cap_hint = match (track.codec_params.n_frames, track.codec_params.channels) {
        (Some(n), Some(c)) => (n as usize).saturating_mul(c.count()),
        _ => 0,
    };
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| DecodeError::Unsupported(e.to_string()))?;

    let mut samples: Vec<f32> = Vec::with_capacity(cap_hint);
    let mut channels = 0usize;
    let mut sample_rate = 0u32;
    let mut sbuf: Option<SampleBuffer<f32>> = None;

    loop {
        if cancel.load(Ordering::Relaxed) { return Err(DecodeError::Cancelled); }
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(DecodeError::Unsupported(e.to_string())),
        };
        if packet.track_id() != track_id { continue; }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = *buf.spec();
                if channels == 0 {
                    channels = spec.channels.count();
                    sample_rate = spec.rate;
                }
                if sbuf.as_ref().map(|b| b.capacity() < buf.capacity() * spec.channels.count()).unwrap_or(true) {
                    sbuf = Some(SampleBuffer::<f32>::new(buf.capacity() as u64, spec));
                }
                let sb = sbuf.as_mut().unwrap();
                sb.copy_interleaved_ref(buf);
                samples.extend_from_slice(sb.samples());
            }
            Err(SymError::DecodeError(_)) => continue, // skip a bad packet, keep going
            Err(e) => return Err(DecodeError::Unsupported(e.to_string())),
        }
    }

    if samples.is_empty() || channels == 0 { return Err(DecodeError::Empty); }
    samples.shrink_to_fit();
    Ok(DecodedAudio { samples, channels, sample_rate, meta })
}

/// Cheap metadata + duration without decoding the audio (used when files are imported).
/// Duration is `None` for containers that do not declare a frame count.
pub fn probe_file(path: &Path) -> Result<(TrackMeta, Option<f32>), DecodeError> {
    let file = File::open(path).map_err(|e| DecodeError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) { hint.with_extension(ext); }
    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DecodeError::Unsupported(e.to_string()))?;
    let mut meta = TrackMeta::default();
    if let Some(md) = probed.metadata.get() {
        if let Some(rev) = md.current() { read_rev(rev, &mut meta); }
    }
    let mut format = probed.format;
    if let Some(rev) = format.metadata().current() { read_rev(rev, &mut meta); }
    if meta.title.is_empty() {
        meta.title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    }
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::Unsupported("no audio track".into()))?;
    let p = &track.codec_params;
    let dur = match (p.n_frames, p.sample_rate) {
        (Some(n), Some(sr)) if sr > 0 => Some(n as f32 / sr as f32),
        _ => None,
    };
    Ok((meta, dur))
}

// ── ffmpeg-based decoding: video files, and audio formats symphonia lacks (Opus, WMA, AC-3 ...) ──

pub fn is_video_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v" | "wmv" | "flv" | "mpg" | "mpeg" | "ts" | "m2ts")
    )
}

/// Length from `ffmpeg -i` (it prints "Duration: HH:MM:SS.xx" on stderr).
pub fn ffmpeg_duration(ffmpeg: &Path, path: &Path) -> Option<f32> {
    let mut c = std::process::Command::new(ffmpeg);
    c.args(["-hide_banner", "-i"]).arg(path)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
    let out = crate::export::spawn_hidden(&mut c).output().ok()?;
    let text = String::from_utf8_lossy(&out.stderr);
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("Duration:") {
            let t = rest.trim().split(',').next()?.trim();
            let p: Vec<f32> = t.split(':').filter_map(|x| x.parse().ok()).collect();
            if p.len() == 3 { return Some(p[0] * 3600.0 + p[1] * 60.0 + p[2]); }
        }
    }
    None
}

/// Decode any audio stream through ffmpeg as stereo 48 kHz f32.
pub fn decode_with_ffmpeg(ffmpeg: &Path, path: &Path, cancel: &AtomicBool) -> Result<DecodedAudio, DecodeError> {
    use std::io::Read;
    let mut c = std::process::Command::new(ffmpeg);
    c.args(["-v", "error", "-nostdin", "-i"]).arg(path)
        .args(["-vn", "-sn", "-dn", "-ac", "2", "-ar", "48000", "-f", "f32le", "pipe:1"])
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    let mut child = crate::export::spawn_hidden(&mut c).spawn().map_err(|e| DecodeError::Io(format!("cannot start ffmpeg: {e}")))?;
    let mut out = child.stdout.take().ok_or_else(|| DecodeError::Io("no ffmpeg output".into()))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut carry: Vec<u8> = Vec::with_capacity(4); // a float can be split across two reads
    loop {
        if cancel.load(Ordering::Relaxed) { let _ = child.kill(); let _ = child.wait(); return Err(DecodeError::Cancelled); }
        let n = match out.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(e) => { let _ = child.kill(); return Err(DecodeError::Io(e.to_string())); } };
        let mut data = &buf[..n];
        while !carry.is_empty() && carry.len() < 4 && !data.is_empty() { carry.push(data[0]); data = &data[1..]; }
        if carry.len() == 4 { samples.push(f32::from_le_bytes([carry[0], carry[1], carry[2], carry[3]])); carry.clear(); }
        let whole = data.len() / 4 * 4;
        for ch in data[..whole].chunks_exact(4) { samples.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]])); }
        carry.extend_from_slice(&data[whole..]);
    }
    let _ = child.wait();
    if samples.len() < 4 { return Err(DecodeError::Unsupported("the file has no audio stream".into())); }
    samples.shrink_to_fit();
    let meta = TrackMeta { title: path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string(), ..Default::default() };
    Ok(DecodedAudio { samples, channels: 2, sample_rate: 48_000, meta })
}

/// Decode with symphonia; video files (and anything symphonia cannot read) go through ffmpeg if available.
pub fn decode_any(path: &Path, ffmpeg: Option<&Path>, cancel: &AtomicBool) -> Result<DecodedAudio, DecodeError> {
    if is_video_path(path) {
        return match ffmpeg {
            Some(f) => decode_with_ffmpeg(f, path, cancel),
            None => Err(DecodeError::Unsupported("video files need ffmpeg (see the Render panel)".into())),
        };
    }
    match decode_file(path, cancel) {
        Err(DecodeError::Unsupported(msg)) => match ffmpeg {
            Some(f) => decode_with_ffmpeg(f, path, cancel).map_err(|_| DecodeError::Unsupported(msg)),
            None => Err(DecodeError::Unsupported(msg)),
        },
        r => r,
    }
}

/// Metadata and length without decoding (ffmpeg for video / unsupported audio).
pub fn probe_any(path: &Path, ffmpeg: Option<&Path>) -> Result<(TrackMeta, Option<f32>), String> {
    let stem_meta = || TrackMeta { title: path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string(), ..Default::default() };
    if is_video_path(path) {
        let f = ffmpeg.ok_or_else(|| "video files need ffmpeg (see the Render panel)".to_string())?;
        return Ok((stem_meta(), ffmpeg_duration(f, path)));
    }
    match probe_file(path) {
        Ok(r) => Ok(r),
        Err(e) => match ffmpeg {
            Some(f) => ffmpeg_duration(f, path).map(|d| (stem_meta(), Some(d))).ok_or_else(|| e.to_string()),
            None => Err(e.to_string()),
        },
    }
}
