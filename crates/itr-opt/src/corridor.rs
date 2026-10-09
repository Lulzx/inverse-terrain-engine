//! Flow corridors and corridor-native placement (spec §7.11.4).
//!
//! From the baseline's time-integrated discharge `Q(x) = ∫ q dt` (sampled at sync points),
//! trace upstream from each protected / guard area along the integrated flux direction.
//! The result is a few corridor polylines that carry water into the targets, ranked by
//! delivered volume, clipped to the editable area. An O(cells) post-process, no extra
//! simulation.
//!
//! Corridor placement re-parameterizes each primitive by arc length `s` along its
//! corridor, a lateral offset, its length and height; width is fixed at the middle of its
//! bounds and orientation follows the local flow (berms across it, swales/cuts along it).
//! A linear primitive drops from 6 to 4 parameters, a mound from 4 to 3 (declared
//! search-space change; `free_placement_fraction` keeps sampling free placements).

use crate::rng::Rand;
use itr_core::design::{Primitive, PrimitiveKind, PrimitiveSpec};
use itr_core::model::{AbortReason, Observer, SyncView};
use itr_core::raster::{GeoTransform, Mask};
use itr_core::scenario::QuantizeCfg;
use serde::Serialize;
use std::ops::ControlFlow;

/// Observer accumulating the time-integrated discharge vector field (east, north).
pub struct QAccum {
    pub dt: f64,
    pub qx: Vec<f64>,
    pub qy: Vec<f64>,
    bx: Vec<f32>,
    by: Vec<f32>,
}

impl QAccum {
    pub fn new(n: usize, sync_interval: f64) -> Self {
        Self { dt: sync_interval, qx: vec![0.0; n], qy: vec![0.0; n], bx: vec![0.0; n], by: vec![0.0; n] }
    }
}

impl Observer for QAccum {
    fn on_sync(&mut self, v: &SyncView<'_>) -> ControlFlow<AbortReason> {
        v.state.discharge(&mut self.bx, &mut self.by);
        for k in 0..self.qx.len() {
            self.qx[k] += self.bx[k] as f64 * self.dt;
            self.qy[k] += self.by[k] as f64 * self.dt;
        }
        ControlFlow::Continue(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Corridor {
    /// World-coordinate polyline, upstream → target.
    pub pts: Vec<(f64, f64)>,
    pub cum: Vec<f64>,
    /// Delivered volume proxy at the target end (m³ per m of width, time-integrated).
    pub volume: f64,
    pub target: String,
}

impl Corridor {
    fn new(pts: Vec<(f64, f64)>, volume: f64, target: String) -> Self {
        let mut cum = vec![0.0];
        for w in pts.windows(2) {
            let d = ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
            cum.push(cum.last().unwrap() + d);
        }
        Self { pts, cum, volume, target }
    }
    pub fn length(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }
    /// Point and unit tangent at arc-length fraction `t ∈ [0,1]`.
    pub fn at(&self, t: f64) -> ((f64, f64), (f64, f64)) {
        let s = t.clamp(0.0, 1.0) * self.length();
        let k = self.cum.partition_point(|&c| c <= s).clamp(1, self.pts.len() - 1);
        let (a, b) = (self.pts[k - 1], self.pts[k]);
        let seg = self.cum[k] - self.cum[k - 1];
        let f = if seg > 0.0 { (s - self.cum[k - 1]) / seg } else { 0.0 };
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let n = (dx * dx + dy * dy).sqrt().max(1e-12);
        ((a.0 + f * dx, a.1 + f * dy), (dx / n, dy / n))
    }
}

/// Extract up to `max` corridors. `qx`/`qy` are time-integrated discharges (east,
/// north-positive) on the fine grid.
#[allow(clippy::too_many_arguments)]
pub fn extract(qx: &[f64], qy: &[f64], nx: usize, ny: usize, geo: &GeoTransform, targets: &[(String, Mask)], editable: &Mask, max: usize) -> Vec<Corridor> {
    let mag = |k: usize| (qx[k] * qx[k] + qy[k] * qy[k]).sqrt();
    // Flow of cell n toward cell c (grid offsets: +col east, +row south → north = −row).
    let toward = |n: usize, c: usize| {
        let (dc, dr) = ((c % nx) as f64 - (n % nx) as f64, (c / nx) as f64 - (n / nx) as f64);
        let (ex, ey) = (dc, -dr);
        let l = (ex * ex + ey * ey).sqrt();
        (qx[n] * ex + qy[n] * ey) / l
    };
    let neigh = |c: usize| {
        let (cc, rr) = ((c % nx) as isize, (c / nx) as isize);
        let mut v = Vec::with_capacity(8);
        for dr in -1..=1isize {
            for dc in -1..=1isize {
                let (a, b) = (cc + dc, rr + dr);
                if (dc, dr) != (0, 0) && a >= 0 && b >= 0 && (a as usize) < nx && (b as usize) < ny {
                    v.push(b as usize * nx + a as usize);
                }
            }
        }
        v
    };
    let mut out: Vec<Corridor> = vec![];
    let mut used = Mask::new(nx, ny, false);
    let mut starts: Vec<(f64, usize, usize, String)> = vec![];
    for (name, m) in targets {
        for k in 0..nx * ny {
            if m.data[k] {
                continue;
            }
            let into: f64 = neigh(k).into_iter().filter(|&t| m.data[t]).map(|t| toward(k, t)).fold(0.0, f64::max);
            if into > 0.0 {
                starts.push((into, k, 0, name.clone()));
            }
        }
    }
    starts.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    for (score0, k0, _, name) in starts {
        if out.len() >= max {
            break;
        }
        if used.data[k0] {
            continue;
        }
        // Trace upstream.
        let mut path = vec![k0];
        let mut seen = std::collections::HashSet::from([k0]);
        let mut cur = k0;
        for _ in 0..4 * (nx + ny) {
            let best = neigh(cur)
                .into_iter()
                .filter(|n| !seen.contains(n) && !targets.iter().any(|(_, m)| m.data[*n]))
                .map(|n| (toward(n, cur), n))
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            match best {
                Some((s, n)) if s > 0.05 * score0 && mag(n) > 0.0 => {
                    path.push(n);
                    seen.insert(n);
                    cur = n;
                }
                _ => break,
            }
        }
        for &k in &path {
            for n in neigh(k) {
                used.data[n] = true;
            }
            used.data[k] = true;
        }
        // Clip to the editable area: longest contiguous editable run.
        let (mut best, mut run) = ((0, 0), (0, 0));
        for (i, &k) in path.iter().enumerate() {
            if editable.data[k] {
                if run.1 == i {
                    run.1 = i + 1;
                } else {
                    run = (i, i + 1);
                }
                if run.1 - run.0 > best.1 - best.0 {
                    best = run;
                }
            } else {
                run = (i + 1, i + 1);
            }
        }
        if best.1 - best.0 < 3 {
            continue;
        }
        let pts: Vec<(f64, f64)> = path[best.0..best.1].iter().rev().map(|&k| geo.cell_center(k % nx, k / nx)).collect();
        out.push(Corridor::new(pts, score0, name));
    }
    out
}

/// Corridor-native design space.
#[derive(Clone, Debug)]
pub struct CorridorSpace {
    pub corridors: Vec<Corridor>,
    pub specs: Vec<PrimitiveSpec>,
    pub quant: QuantizeCfg,
}

fn linear(k: PrimitiveKind) -> bool {
    !matches!(k, PrimitiveKind::Mound)
}

impl CorridorSpace {
    pub fn dimension(&self) -> usize {
        self.specs.iter().map(|s| if linear(s.kind) { 4 } else { 3 }).sum()
    }
    pub fn param_names(&self) -> Vec<String> {
        let mut v = vec![];
        for (k, s) in self.specs.iter().enumerate() {
            let names: &[&str] = if linear(s.kind) { &["s", "offset", "length", "height"] } else { &["s", "offset", "height"] };
            v.extend(names.iter().map(|n| format!("p{k}.{n}")));
        }
        v
    }
    fn q(v: f64, step: f64) -> f64 {
        if step > 0.0 { (v / step).round() * step } else { v }
    }
    pub fn decode(&self, u: &[f64]) -> Vec<Primitive> {
        let lerp = |b: [f64; 2], t: f64| b[0] + (b[1] - b[0]) * t.clamp(0.0, 1.0);
        let qc = &self.quant;
        let nc = self.corridors.len().max(1);
        let mut out = vec![];
        let mut k = 0;
        for (i, s) in self.specs.iter().enumerate() {
            let c = &self.corridors[i % nc];
            let ((px, py), (tx, ty)) = c.at(u[k]);
            let width = Self::q(0.5 * (s.width_m[0] + s.width_m[1]), qc.width_m);
            let half = 1.5 * s.width_m[1];
            let off = lerp([-half, half], u[k + 1]);
            let (cx, cy) = (Self::q(px - ty * off, qc.position_m), Self::q(py + tx * off, qc.position_m));
            let tang = ty.atan2(tx).to_degrees();
            let (length, angle, hk) = if linear(s.kind) {
                let ang = if s.kind == PrimitiveKind::Berm { tang + 90.0 } else { tang };
                (Self::q(lerp(s.length_m, u[k + 2]), qc.position_m), Self::q(ang.rem_euclid(180.0), qc.angle_deg), k + 3)
            } else {
                (0.0, 0.0, k + 2)
            };
            let sign = if matches!(s.kind, PrimitiveKind::Berm | PrimitiveKind::Mound) { 1.0 } else { -1.0 };
            let height = sign * Self::q(lerp(s.height_m, u[hk]), qc.height_m);
            k = hk + 1;
            out.push(Primitive { kind: s.kind, cx, cy, length, width: width.max(1e-6), angle_deg: angle, height });
        }
        out
    }
    /// "Berm across the top-ranked corridor" seed (§7.11.4): the first berm sits mid-way
    /// along corridor 0 at 80% height; other primitives start at zero height.
    pub fn heuristic_mean(&self, s_pos: f64) -> Vec<f64> {
        let mut u = vec![];
        let mut first = true;
        for s in &self.specs {
            let lin = linear(s.kind);
            let on = first && s.kind == PrimitiveKind::Berm;
            first &= !on;
            u.push(s_pos);
            u.push(0.5);
            if lin {
                u.push(0.6);
            }
            u.push(if on { 0.8 } else { 0.0 });
        }
        u
    }
    /// Physics-informed restart means: the seed berm at other positions along the corridor.
    pub fn restart_means(&self) -> Vec<Vec<f64>> {
        [0.25, 0.75, 0.4, 0.6].iter().map(|&s| self.heuristic_mean(s)).collect()
    }
}

/// Free-placement proposals mixed into corridor-mode generations.
pub fn free_proposals(n: usize, count: usize, rng: &mut Rand) -> Vec<Vec<f64>> {
    (0..count).map(|_| (0..n).map(|_| rng.uniform()).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traces_a_straight_channel_upstream() {
        // Uniform southward flow in column 5 into a target at the bottom.
        let (nx, ny) = (11, 20);
        let mut qx = vec![0.0; nx * ny];
        let mut qy = vec![0.0; nx * ny];
        for r in 0..ny {
            qy[r * nx + 5] = -10.0;
            for c in [4, 6] {
                qy[r * nx + c] = -1.0;
            }
        }
        let _ = &mut qx;
        let geo = GeoTransform { origin_x: 0.0, origin_y: ny as f64, dx: 1.0, dy: 1.0 };
        let mut target = Mask::new(nx, ny, false);
        for c in 3..8 {
            target.set(c, ny - 1, true);
        }
        let editable = Mask::new(nx, ny, true);
        let cs = extract(&qx, &qy, nx, ny, &geo, &[("house".into(), target)], &editable, 2);
        assert!(!cs.is_empty());
        let c = &cs[0];
        assert!(c.pts.len() > 10, "corridor too short: {}", c.pts.len());
        // Runs along column 5 (x = 5.5), upstream (north) → target (south).
        assert!(c.pts.iter().all(|p| (p.0 - 5.5).abs() < 1e-9));
        assert!(c.pts.first().unwrap().1 > c.pts.last().unwrap().1);
        // A berm placed by the corridor space lies across the flow (east–west).
        let spec = PrimitiveSpec { kind: PrimitiveKind::Berm, height_m: [0.0, 1.0], width_m: [2.0, 4.0], length_m: [4.0, 8.0], angle_deg: [0.0, 180.0], x_m: None, y_m: None };
        let space = CorridorSpace { corridors: cs, specs: vec![spec], quant: QuantizeCfg::default() };
        let p = &space.decode(&[0.5, 0.5, 0.5, 1.0])[0];
        assert!((p.angle_deg.rem_euclid(180.0) - 0.0).abs() < 1.0 || (p.angle_deg - 180.0).abs() < 1.0, "angle {}", p.angle_deg);
        assert_eq!(space.dimension(), 4);
    }
}
