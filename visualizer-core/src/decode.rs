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
