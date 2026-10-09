//! Run data written by `scripts/video_data.py`: metadata and packed textures.
use fframes::media::{ImageData, ImageMetadata};
use fframes::usvgr::PreloadedImageData;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub cell: f32,
    pub up: usize,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Deserialize)]
pub struct Elevation {
    pub lo: f32,
    pub span: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Deserialize)]
pub struct House {
    pub name: String,
    pub threshold: f32,
    pub xy: Vec<[f32; 2]>,
}

#[derive(Debug, Deserialize)]
pub struct Ring {
    pub xy: Vec<[f32; 2]>,
}

#[derive(Debug, Deserialize)]
pub struct Outcome {
    pub j_before: f64,
    pub j_after: f64,
    pub guard_worsening_m: f64,
    pub guard_tolerance_m: f64,
    pub fill_m3: f64,
    pub cut_m3: f64,
}

#[derive(Debug, Deserialize)]
pub struct Meta {
    pub example: String,
    pub grid: Grid,
    pub z: Elevation,
    pub vel_max: f32,
    /// Water depth of texture code 65535 (m).
    pub depth_span: f32,
    pub sync_s: f32,
    pub times: Vec<f32>,
    pub rain: Vec<f32>,
    pub rain_max: f32,
    pub houses: Vec<House>,
    pub guards: Vec<Ring>,
    pub earthworks: Vec<Ring>,
    /// "before" / "after" -> per house, max depth on its footprint per frame (m).
    pub series: HashMap<String, Vec<Vec<f32>>>,
    pub result: Outcome,
    pub manning_n: f32,
}

/// One terrain variant (no change or design) with its per-frame water state.
pub struct View {
    pub terrain: ImageData<'static>,
    pub depth: Vec<ImageData<'static>>,
    pub vel: Vec<ImageData<'static>>,
    /// Depth at each house per frame (m).
    pub series: Vec<Vec<f32>>,
    /// Roof-top point of each house in world coordinates (m, true scale above `z.lo`).
    pub house_tops: Vec<[f32; 3]>,
}

pub struct RunData {
    pub meta: Meta,
    pub views: [View; 2],
    /// Mean ground elevation above `z.lo` (m).
    pub ground_mean: f32,
}

fn texture(id: String, w: usize, h: usize, bytes: Vec<u8>) -> ImageData<'static> {
    let pixels = PreloadedImageData { data: Cow::Owned(bytes), width: w as u32, height: h as u32, id: id.clone() };
    ImageData::new_from_raw_data(Arc::new(pixels), id, ImageMetadata { width: w as u32, height: h as u32 })
}

fn decode16(px: &[u8]) -> f32 {
    (px[0] as f32 * 256.0 + px[1] as f32) / 65535.0
}

impl RunData {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
        let meta: Meta = serde_json::from_slice(&read("meta.json")?).map_err(|e| format!("meta.json: {e}"))?;
        let (nx, ny, up) = (meta.grid.nx, meta.grid.ny, meta.grid.up);
        let (fx, fy) = (nx * up, ny * up);
        let n = meta.times.len();
        let frame_bytes = nx * ny * 4;
        let mut ground_mean = 0.0;

        let mut views = Vec::new();
        for tag in ["before", "after"] {
            let tb = read(&format!("terrain_{tag}.rgba"))?;
            if tb.len() != fx * fy * 4 {
                return Err(format!("terrain_{tag}.rgba: unexpected size"));
            }
            let elev = |x: f32, y: f32| {
                let c = ((x / meta.grid.cell * up as f32) as usize).min(fx - 1);
                let r = (((meta.grid.height - y) / meta.grid.cell * up as f32) as usize).min(fy - 1);
                decode16(&tb[(r * fx + c) * 4..]) * meta.z.span
            };
            if tag == "before" {
                ground_mean = (0..fy).step_by(4).flat_map(|r| (0..fx).step_by(4).map(move |c| (r, c)))
                    .map(|(r, c)| decode16(&tb[(r * fx + c) * 4..]) * meta.z.span).sum::<f32>()
                    / ((fy.div_ceil(4) * fx.div_ceil(4)) as f32);
            }
            let house_tops = meta.houses.iter().map(|h| {
                let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for p in &h.xy {
                    x0 = x0.min(p[0]);
                    x1 = x1.max(p[0]);
                    y0 = y0.min(p[1]);
                    y1 = y1.max(p[1]);
                }
                let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
                [cx, cy, elev(cx, cy)]
            }).collect();
            let terrain = texture(format!("terrain_{tag}"), fx, fy, tb);

            let eb = read(&format!("depth_{tag}.rgba"))?;
            let vb = read(&format!("vel_{tag}.rgba"))?;
            if eb.len() != n * frame_bytes || vb.len() != n * frame_bytes {
                return Err(format!("depth/vel_{tag}.rgba: expected {n} frames"));
            }
            let depth = (0..n).map(|i| texture(format!("depth_{tag}_{i}"), nx, ny, eb[i * frame_bytes..(i + 1) * frame_bytes].to_vec())).collect();
            let vel = (0..n).map(|i| texture(format!("vel_{tag}_{i}"), nx, ny, vb[i * frame_bytes..(i + 1) * frame_bytes].to_vec())).collect();
            let series = meta.series.get(tag).cloned().ok_or(format!("series.{tag} missing"))?;
            views.push(View { terrain, depth, vel, series, house_tops });
        }
        let [before, after]: [View; 2] = views.try_into().map_err(|_| "views")?;
        Ok(Self { meta, views: [before, after], ground_mean })
    }

    pub fn frames(&self) -> usize {
        self.meta.times.len()
    }
}
