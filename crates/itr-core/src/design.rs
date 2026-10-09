//! Earthwork design space (spec §6.1–6.2): compact-support primitives, quantization,
//! resolution-independent materialization and static feasibility.

use crate::raster::{GeoTransform, Mask, Raster};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveKind {
    /// Linear raised ridge, smooth C¹ cross-section.
    Berm,
    /// Linear smooth depression.
    Swale,
    /// Linear flat-bottomed cut with smooth side slopes.
    Cut,
    /// Radial raised mound (no length/orientation).
    Mound,
}

impl PrimitiveKind {
    fn sign(self) -> f64 {
        match self {
            PrimitiveKind::Berm | PrimitiveKind::Mound => 1.0,
            PrimitiveKind::Swale | PrimitiveKind::Cut => -1.0,
        }
    }
    fn linear(self) -> bool {
        !matches!(self, PrimitiveKind::Mound)
    }
}

/// One primitive slot in `earthworks.toml`. Bounds are `[min, max]`; centre bounds
/// default to the editable mask's bounding box.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveSpec {
    pub kind: PrimitiveKind,
    pub height_m: [f64; 2],
    pub width_m: [f64; 2],
    #[serde(default = "d_len")]
    pub length_m: [f64; 2],
    #[serde(default = "d_ang")]
    pub angle_deg: [f64; 2],
    #[serde(default)]
    pub x_m: Option<[f64; 2]>,
    #[serde(default)]
    pub y_m: Option<[f64; 2]>,
}
fn d_len() -> [f64; 2] {
    [5.0, 50.0]
}
fn d_ang() -> [f64; 2] {
    [0.0, 180.0]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitivesFile {
    pub primitive: Vec<PrimitiveSpec>,
}

impl PrimitivesFile {
    pub fn from_toml(s: &str) -> Result<Self, String> {
        let f: PrimitivesFile = toml::from_str(s).map_err(|e| e.to_string())?;
        if f.primitive.is_empty() {
            return Err("earthworks file must declare at least one [[primitive]]".into());
        }
        for p in &f.primitive {
            for b in [p.height_m, p.width_m, p.length_m, p.angle_deg] {
                if !(b[0] <= b[1]) {
                    return Err("primitive bounds must satisfy min <= max".into());
                }
            }
            if p.height_m[0] < 0.0 || p.width_m[0] <= 0.0 {
                return Err("height_m must be >= 0 (sign comes from kind) and width_m > 0".into());
            }
        }
        Ok(f)
    }
}

/// A concrete, quantized primitive in world coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    pub cx: f64,
    pub cy: f64,
    pub length: f64,
    pub width: f64,
    pub angle_deg: f64,
    /// Signed height change at the crest/centre (m).
    pub height: f64,
}

impl Primitive {
    /// Segment endpoints (for linear kinds) or the centre twice.
    pub fn segment(&self) -> ((f64, f64), (f64, f64)) {
        if !self.kind.linear() {
            return ((self.cx, self.cy), (self.cx, self.cy));
        }
        let a = self.angle_deg.to_radians();
        let (s, c) = (libm::sin(a), libm::cos(a));
        let hl = 0.5 * self.length;
        ((self.cx - c * hl, self.cy - s * hl), (self.cx + c * hl, self.cy + s * hl))
    }

    /// Profile value in [0,1] at world point; exactly zero outside the support.
    pub fn profile(&self, x: f64, y: f64) -> f64 {
        let ((x0, y0), (x1, y1)) = self.segment();
        let (vx, vy) = (x1 - x0, y1 - y0);
        let l2 = vx * vx + vy * vy;
        let t = if l2 > 0.0 { (((x - x0) * vx + (y - y0) * vy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        let (px, py) = (x0 + t * vx - x, y0 + t * vy - y);
        let d = libm::sqrt(px * px + py * py);
        let s = d / (0.5 * self.width);
        if s >= 1.0 {
            return 0.0;
        }
        match self.kind {
            PrimitiveKind::Cut => {
                if s <= 0.5 {
                    1.0
                } else {
                    let u = (s - 0.5) / 0.5;
                    let w = 1.0 - u * u;
                    w * w
                }
            }
            _ => {
                let w = 1.0 - s * s;
                w * w
            }
        }
    }

    /// Axis-aligned world bounding box of the support.
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let ((x0, y0), (x1, y1)) = self.segment();
        let r = 0.5 * self.width;
        (x0.min(x1) - r, y0.min(y1) - r, x0.max(x1) + r, y0.max(y1) + r)
    }
}

/// Sparse edit: bbox (inclusive-exclusive, cell indices) + Δz values row-major.
#[derive(Clone, Debug, Default)]
pub struct TerrainEdit {
    pub c0: usize,
    pub r0: usize,
    pub w: usize,
    pub h: usize,
    pub dz: Vec<f64>,
    pub cut_m3: f64,
    pub fill_m3: f64,
    pub primitives: Vec<Primitive>,
}

impl TerrainEdit {
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0 || self.dz.iter().all(|v| *v == 0.0)
    }
    pub fn get(&self, col: usize, row: usize) -> f64 {
        if col < self.c0 || row < self.r0 || col >= self.c0 + self.w || row >= self.r0 + self.h {
            return 0.0;
        }
        self.dz[(row - self.r0) * self.w + (col - self.c0)]
    }
    /// Apply to a full raster (used for exports and tests).
    pub fn apply(&self, base: &Raster) -> Raster {
        let mut out = base.clone();
        for r in 0..self.h {
            for c in 0..self.w {
                let i = out.idx(self.c0 + c, self.r0 + r);
                out.data[i] += self.dz[r * self.w + c];
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub name: String,
    /// Positive magnitude of violation in natural units.
    pub amount: f64,
}

/// Continuous design space mapping normalized `u ∈ [0,1]^n` to quantized primitives.
#[derive(Clone, Debug)]
pub struct DesignSpace {
    pub specs: Vec<PrimitiveSpec>,
    pub center_bounds: Vec<([f64; 2], [f64; 2])>,
    pub quant: crate::scenario::QuantizeCfg,
    pub max_abs_dz: f64,
    pub max_volume_m3: f64,
    pub max_edit_slope: f64,
}

impl DesignSpace {
    pub fn new(
        specs: Vec<PrimitiveSpec>,
        editable: &Mask,
        geo: &GeoTransform,
        cfg: &crate::scenario::DesignCfg,
    ) -> Result<Self, String> {
        // Default centre bounds: bbox of editable cell centres.
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for r in 0..editable.ny {
            for c in 0..editable.nx {
                if editable.get(c, r) {
                    let (x, y) = geo.cell_center(c, r);
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                }
            }
        }
        if x0 > x1 {
            return Err("editable mask is empty".into());
        }
        let center_bounds = specs
            .iter()
            .map(|s| (s.x_m.unwrap_or([x0, x1]), s.y_m.unwrap_or([y0, y1])))
            .collect();
        Ok(Self {
            specs,
            center_bounds,
            quant: cfg.quantize.clone(),
            max_abs_dz: cfg.max_abs_elevation_change_m,
            max_volume_m3: cfg.max_earthwork_volume_m3,
            max_edit_slope: cfg.max_edit_slope,
        })
    }

    fn params_of(kind: PrimitiveKind) -> usize {
        if kind.linear() { 6 } else { 4 }
    }

    pub fn dimension(&self) -> usize {
        self.specs.iter().map(|s| Self::params_of(s.kind)).sum()
    }

    /// Human-readable parameter names, in θ order.
    pub fn param_names(&self) -> Vec<String> {
        let mut v = vec![];
        for (k, s) in self.specs.iter().enumerate() {
            let names: &[&str] = if s.kind.linear() {
                &["x", "y", "length", "width", "angle", "height"]
            } else {
                &["x", "y", "width", "height"]
            };
            for n in names {
                v.push(format!("p{k}.{n}"));
            }
        }
        v
    }

    fn q(v: f64, step: f64) -> f64 {
        if step > 0.0 { (v / step).round() * step } else { v }
    }

    /// Map normalized `u` (clamped to [0,1]) to quantized primitives.
    pub fn decode(&self, u: &[f64]) -> Vec<Primitive> {
        assert_eq!(u.len(), self.dimension());
        let lerp = |b: [f64; 2], t: f64| b[0] + (b[1] - b[0]) * t.clamp(0.0, 1.0);
        let qc = &self.quant;
        let mut out = vec![];
        let mut k = 0;
        for (i, s) in self.specs.iter().enumerate() {
            let (bx, by) = self.center_bounds[i];
            let cx = Self::q(lerp(bx, u[k]), qc.position_m);
            let cy = Self::q(lerp(by, u[k + 1]), qc.position_m);
            k += 2;
            let (length, angle_deg) = if s.kind.linear() {
                let l = Self::q(lerp(s.length_m, u[k]), qc.position_m);
                let a = Self::q(lerp(s.angle_deg, u[k + 2]), qc.angle_deg);
                (l, a)
            } else {
                (0.0, 0.0)
            };
            let width = if s.kind.linear() {
                let w = Self::q(lerp(s.width_m, u[k + 1]), qc.width_m);
                k += 3;
                w
            } else {
                let w = Self::q(lerp(s.width_m, u[k]), qc.width_m);
                k += 1;
                w
            };
            let height = s.kind.sign() * Self::q(lerp(s.height_m, u[k]), qc.height_m);
            k += 1;
            out.push(Primitive { kind: s.kind, cx, cy, length, width: width.max(1e-6), angle_deg, height });
        }
        out
    }

    /// Quantized parameter key (for caching): integers of quantized values.
    pub fn cache_key(&self, prims: &[Primitive]) -> Vec<i64> {
        let qc = &self.quant;
        let mut v = vec![];
        let qi = |x: f64, s: f64| if s > 0.0 { (x / s).round() as i64 } else { x.to_bits() as i64 };
        for p in prims {
            v.extend([
                qi(p.cx, qc.position_m),
                qi(p.cy, qc.position_m),
                qi(p.length, qc.position_m),
                qi(p.width, qc.width_m),
                qi(p.angle_deg, qc.angle_deg),
                qi(p.height, qc.height_m),
            ]);
        }
        v
    }

    /// Deterministic materialization at any resolution: 4×4 sub-sample quadrature per
    /// cell, summed over primitives, masked by `editable`, clamped to ±max_abs_dz.
    pub fn materialize(&self, prims: &[Primitive], geo: &GeoTransform, nx: usize, ny: usize, editable: &Mask) -> TerrainEdit {
        let mut edit = TerrainEdit { primitives: prims.to_vec(), ..Default::default() };
        // Union bbox in cells.
        let (mut c0, mut r0, mut c1, mut r1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for p in prims {
            if p.height == 0.0 {
                continue;
            }
            let (x0, y0, x1, y1) = p.bbox();
            let (fc0, fr1) = geo.to_grid(x0, y0);
            let (fc1, fr0) = geo.to_grid(x1, y1);
            let a = (fc0.floor().max(0.0) as usize).min(nx);
            let b = (fr0.floor().max(0.0) as usize).min(ny);
            let c = ((fc1.ceil().max(0.0)) as usize).min(nx);
            let d = ((fr1.ceil().max(0.0)) as usize).min(ny);
            if a < c && b < d {
                c0 = c0.min(a);
                r0 = r0.min(b);
                c1 = c1.max(c);
                r1 = r1.max(d);
            }
        }
        if c0 >= c1 || r0 >= r1 {
            return edit;
        }
        let (w, h) = (c1 - c0, r1 - r0);
        let mut dz = vec![0.0; w * h];
        const S: usize = 4;
        let inv = 1.0 / (S * S) as f64;
        for r in 0..h {
            for c in 0..w {
                let (col, row) = (c0 + c, r0 + r);
                if !editable.get(col, row) {
                    continue;
                }
                let mut acc = 0.0;
                for p in prims {
                    if p.height == 0.0 {
                        continue;
                    }
                    let mut s = 0.0;
                    for sy in 0..S {
                        for sx in 0..S {
                            let x = geo.origin_x + (col as f64 + (sx as f64 + 0.5) / S as f64) * geo.dx;
                            let y = geo.origin_y - (row as f64 + (sy as f64 + 0.5) / S as f64) * geo.dy;
                            s += p.profile(x, y);
                        }
                    }
                    acc += p.height * s * inv;
                }
                dz[r * w + c] = acc.clamp(-self.max_abs_dz, self.max_abs_dz);
            }
        }
        let a = geo.cell_area();
        for v in &dz {
            if *v > 0.0 {
                edit.fill_m3 += v * a;
            } else {
                edit.cut_m3 -= v * a;
            }
        }
        edit.c0 = c0;
        edit.r0 = r0;
        edit.w = w;
        edit.h = h;
        edit.dz = dz;
        edit
    }

    /// Simulation-free checks in O(edit bbox) (spec §6.2).
    pub fn static_violations(&self, edit: &TerrainEdit, base: &Raster, out: &mut Vec<ConstraintViolation>) {
        let vol = edit.cut_m3 + edit.fill_m3;
        if vol > self.max_volume_m3 {
            out.push(ConstraintViolation { name: "earthwork_volume".into(), amount: vol - self.max_volume_m3 });
        }
        if edit.w == 0 {
            return;
        }
        let g = &base.geo;
        if self.max_edit_slope > 0.0 {
            let mut worst: f64 = 0.0;
            for r in 0..edit.h {
                for c in 0..edit.w {
                    let v = edit.dz[r * edit.w + c];
                    if c + 1 < edit.w {
                        worst = worst.max((edit.dz[r * edit.w + c + 1] - v).abs() / g.dx);
                    }
                    if r + 1 < edit.h {
                        worst = worst.max((edit.dz[(r + 1) * edit.w + c] - v).abs() / g.dy);
                    }
                }
            }
            if worst > self.max_edit_slope {
                out.push(ConstraintViolation { name: "edit_slope".into(), amount: worst - self.max_edit_slope });
            }
        }
        // New single-cell pits: edited cell strictly below all 8 neighbours by > 1 cm
        // where the original was not.
        let z = |c: usize, r: usize, edited: bool| base.get(c, r) + if edited { edit.get(c, r) } else { 0.0 };
        let mut pits = 0usize;
        for r in edit.r0.max(1)..(edit.r0 + edit.h).min(base.ny - 1) {
            for c in edit.c0.max(1)..(edit.c0 + edit.w).min(base.nx - 1) {
                if edit.get(c, r) >= 0.0 {
                    continue;
                }
                let is_pit = |ed: bool| {
                    let zc = z(c, r, ed);
                    (r - 1..=r + 1).all(|rr| (c - 1..=c + 1).all(|cc| (rr == r && cc == c) || z(cc, rr, ed) > zc + 0.01))
                };
                if is_pit(true) && !is_pit(false) {
                    pits += 1;
                }
            }
        }
        if pits > 0 {
            out.push(ConstraintViolation { name: "new_pit".into(), amount: pits as f64 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::DesignCfg;

    fn cfg() -> DesignCfg {
        toml::from_str(
            r#"editable_mask = "m"
primitives = "p"
max_abs_elevation_change_m = 0.75
max_earthwork_volume_m3 = 1e9"#,
        )
        .unwrap()
    }

    #[test]
    fn compact_support_and_volume_is_cell_sum() {
        let geo = GeoTransform { origin_x: 0.0, origin_y: 100.0, dx: 1.0, dy: 1.0 };
        let m = Mask::new(100, 100, true);
        let spec = PrimitiveSpec {
            kind: PrimitiveKind::Berm,
            height_m: [0.5, 0.5],
            width_m: [4.0, 4.0],
            length_m: [20.0, 20.0],
            angle_deg: [0.0, 0.0],
            x_m: Some([50.0, 50.0]),
            y_m: Some([50.0, 50.0]),
        };
        let ds = DesignSpace::new(vec![spec], &m, &geo, &cfg()).unwrap();
        let p = ds.decode(&[0.5; 6]);
        let e = ds.materialize(&p, &geo, 100, 100, &m);
        let sum: f64 = e.dz.iter().sum();
        assert!((sum - e.fill_m3).abs() < 1e-9);
        // Outside bbox exactly zero by construction; bbox no larger than support + 1 cell.
        assert!(e.w <= 26 && e.h <= 6, "w={} h={}", e.w, e.h);
        // analytic volume of (1-s²)² ridge cross-section: area = w/2 * 2 * 8/15 * h
        let section = 0.5 * 4.0 * 2.0 * 8.0 / 15.0 * 0.5;
        assert!((e.fill_m3 / (section * 20.0) - 1.0).abs() < 0.2);
    }

    #[test]
    fn masked_cells_untouched() {
        let geo = GeoTransform { origin_x: 0.0, origin_y: 50.0, dx: 1.0, dy: 1.0 };
        let mut m = Mask::new(50, 50, true);
        for r in 0..50 {
            for c in 0..25 {
                m.set(c, r, false);
            }
        }
        let spec = PrimitiveSpec {
            kind: PrimitiveKind::Mound,
            height_m: [1.0, 1.0],
            width_m: [10.0, 10.0],
            length_m: [0.0, 0.0],
            angle_deg: [0.0, 0.0],
            x_m: Some([25.0, 25.0]),
            y_m: Some([25.0, 25.0]),
        };
        let ds = DesignSpace::new(vec![spec], &m, &geo, &cfg()).unwrap();
        let e = ds.materialize(&ds.decode(&[0.5; 4]), &geo, 50, 50, &m);
        for r in 0..50 {
            for c in 0..25 {
                assert_eq!(e.get(c, r), 0.0);
            }
        }
        assert!(e.dz.iter().cloned().fold(0.0, f64::max) <= 0.75);
    }
}
