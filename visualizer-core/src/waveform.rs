//! Min/max peak pyramid for drawing a waveform at any zoom without rescanning samples.

const BASE: usize = 256; // frames per level-0 block; each level doubles it

pub struct Waveform {
    pub sample_rate: u32,
    pub duration_s: f32,
    levels: Vec<Vec<(f32, f32)>>,
}

impl Waveform {
    pub fn build(samples: &[f32], channels: usize, sample_rate: u32) -> Self {
        let ch = channels.max(1);
        let frames = samples.len() / ch;
        let mut l0 = Vec::with_capacity(frames / BASE + 1);
        for block in samples.chunks(BASE * ch) {
            let (mut mn, mut mx) = (f32::MAX, f32::MIN);
            for fr in block.chunks_exact(ch) {
                let v = fr.iter().sum::<f32>() / ch as f32;
                mn = mn.min(v);
                mx = mx.max(v);
            }
            if mn > mx { mn = 0.0; mx = 0.0; }
            l0.push((mn, mx));
        }
        let mut levels = vec![l0];
        while levels.last().map(|l| l.len()).unwrap_or(0) > 1 {
            let prev = levels.last().unwrap();
            let next: Vec<(f32, f32)> = prev.chunks(2)
                .map(|c| c.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.0), a.1.max(p.1))))
                .collect();
            levels.push(next);
        }
        Self { sample_rate, duration_s: frames as f32 / sample_rate.max(1) as f32, levels }
    }

    /// `buckets` (min, max) pairs covering [t0, t1] seconds.
    pub fn peaks(&self, t0: f32, t1: f32, buckets: usize) -> Vec<(f32, f32)> {
        let buckets = buckets.max(1);
        let sr = self.sample_rate as f64;
        let span = ((t1 - t0).max(0.0) as f64 * sr) / buckets as f64; // frames per bucket
        let mut lvl = 0usize;
        while lvl + 1 < self.levels.len() && ((BASE << (lvl + 1)) as f64) <= span { lvl += 1; }
        let block = (BASE << lvl) as f64;
        let data = &self.levels[lvl];
        (0..buckets).map(|b| {
            let s0 = t0 as f64 * sr + b as f64 * span;
            let i0 = (s0 / block).floor().max(0.0) as usize;
            let i1 = (((s0 + span) / block).ceil() as usize).max(i0 + 1);
            let mut acc = (f32::MAX, f32::MIN);
            for i in i0..i1.min(data.len()) {
                acc = (acc.0.min(data[i].0), acc.1.max(data[i].1));
            }
            if acc.0 > acc.1 { (0.0, 0.0) } else { acc }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn peaks_cover_range() {
        let s: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin()).collect();
        let w = Waveform::build(&s, 1, 48_000);
        let p = w.peaks(0.0, 1.0, 300);
        assert_eq!(p.len(), 300);
        assert!(p.iter().any(|&(mn, mx)| mx > 0.5 && mn < -0.5));
    }
}
