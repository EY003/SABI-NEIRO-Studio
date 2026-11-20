//! Background-video decoding through an ffmpeg child process (rawvideo RGBA on stdout).
//! No ffprobe needed: `ffmpeg -i file` prints the stream info on stderr and we parse that.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};

use visualizer_core::export::spawn_hidden;

#[derive(Clone, Copy, Debug)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub duration_s: Option<f64>,
}

pub fn parse_probe(stderr: &str) -> Option<VideoInfo> {
    let mut dur = None;
    let mut dims = None;
    for line in stderr.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("Duration:") {
            let t = rest.trim().split(',').next().unwrap_or("").trim();
            let p: Vec<f64> = t.split(':').filter_map(|x| x.parse().ok()).collect();
            if p.len() == 3 { dur = Some(p[0] * 3600.0 + p[1] * 60.0 + p[2]); }
        }
        if dims.is_none() && l.contains("Video:") {
            // tokens like "1920x1080" or "1920x1080," or "1920x1080 [SAR 1:1 ...]"
            for tok in l.split(|c: char| c == ' ' || c == ',' || c == '[') {
                if let Some((a, b)) = tok.split_once('x') {
                    if let (Ok(w), Ok(h)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        if w >= 16 && h >= 16 { dims = Some((w, h)); break; }
                    }
                }
            }
        }
    }
    dims.map(|(w, h)| VideoInfo { width: w, height: h, duration_s: dur })
}

pub fn probe(ffmpeg: &Path, file: &Path) -> Result<VideoInfo, String> {
    let mut c = Command::new(ffmpeg);
    c.args(["-hide_banner", "-i"]).arg(file).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    let out = spawn_hidden(&mut c).output().map_err(|e| format!("Cannot run ffmpeg: {e}"))?;
    let text = String::from_utf8_lossy(&out.stderr);
    parse_probe(&text).ok_or_else(|| format!("Not a readable video: {}", file.display()))
}

/// Even output size capped at `max_w` (keeps aspect).
pub fn scaled_size(w: usize, h: usize, max_w: usize) -> (usize, usize) {
    let s = (max_w as f32 / w as f32).min(1.0);
    let even = |v: f32| (((v.round() as usize).max(2)) / 2) * 2;
    (even(w as f32 * s), even(h as f32 * s))
}

pub struct VideoStream {
    rx: Receiver<Vec<u8>>,
    child: Child,
    done: bool,
    pub w: usize,
    pub h: usize,
}

impl VideoStream {
    /// Frames come out at exactly `fps`, starting at media time `start_s`.
    pub fn open(ffmpeg: &Path, file: &Path, start_s: f64, fps: f32, w: usize, h: usize, hwdec: bool) -> Result<Self, String> {
        let mut c = Command::new(ffmpeg);
        c.args(["-v", "error"]);
        if hwdec { c.args(["-hwaccel", "auto"]); } // D3D11/DXVA/CUDA when available, else software
        c.arg("-ss").arg(format!("{start_s:.4}")).arg("-i").arg(file)
            .args(["-an", "-sn", "-vf"]).arg(format!("fps={fps:.5},scale={w}:{h}:flags=bicubic"))
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = spawn_hidden(&mut c).spawn().map_err(|e| format!("Cannot start ffmpeg: {e}"))?;
        let mut out = child.stdout.take().ok_or("no ffmpeg stdout")?;
        let (tx, rx) = sync_channel::<Vec<u8>>(3);
        let n = w * h * 4;
        std::thread::spawn(move || {
            loop {
                let mut buf = vec![0u8; n];
                if out.read_exact(&mut buf).is_err() { break; }
                if tx.send(buf).is_err() { break; }
            }
            // dropping `tx` disconnects the channel once the queue is drained
        });
        Ok(Self { rx, child, done: false, w, h })
    }

    pub fn try_next(&mut self) -> Option<Vec<u8>> {
        match self.rx.try_recv() {
            Ok(b) => Some(b),
            Err(TryRecvError::Disconnected) => { self.done = true; None }
            Err(TryRecvError::Empty) => None,
        }
    }

    /// The decoder finished and every frame has been taken (only true after `try_next` saw it).
    pub fn is_finished(&self) -> bool { self.done }
}

impl Drop for VideoStream {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_ffmpeg_banner() {
        let s = "Input #0, mov,mp4, from 'a.mp4':\n  Duration: 00:01:05.50, start: 0.000000, bitrate: 5000 kb/s\n  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 4900 kb/s, 30 fps\n";
        let i = parse_probe(s).unwrap();
        assert_eq!((i.width, i.height), (1920, 1080));
        assert!((i.duration_s.unwrap() - 65.5).abs() < 1e-6);
    }
    #[test]
    fn sizes_are_even_and_capped() {
        assert_eq!(scaled_size(1920, 1080, 960), (960, 540));
        assert_eq!(scaled_size(640, 480, 960), (640, 480));
    }
}
