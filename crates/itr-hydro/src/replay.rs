//! Subdomain replay (spec §7.11.3). The baseline run (with locked Δt) records, every step,
//! the 1-cell ghost ring just outside a region of interest (ROI) and, at sync points, the
//! 1-cell band just inside it. Candidates then run on the ROI only, with the recorded ring
//! as time-dependent ghost cells. Boundary faces are evaluated by the same face function
//! with the same inputs as in the full domain, so a zero-edit replay is bitwise identical
//! to the baseline restricted to the ROI. Any edit influence reaching the band is detected
//! at sync points and the run is marked replay-invalid (escalated by the caller).

use crate::aligned::Layout;
use crate::solver::Checkpoint;
use crate::terrain::{Boundaries, FaceRows, PreparedTerrain, RowTable, TerrainBase};
use itr_core::raster::{GeoTransform, Mask};
use itr_core::Real;
use std::sync::Arc;

/// ROI in level-grid cell coordinates (0-based, interior), half-open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub c0: usize,
    pub r0: usize,
    pub w: usize,
    pub h: usize,
}

impl Roi {
    /// Padded ROI-layout coordinates of the ghost ring, in a fixed order.
    pub fn ring(&self) -> Vec<(usize, usize)> {
        let (w, h) = (self.w, self.h);
        let mut v = Vec::with_capacity(2 * (w + 2) + 2 * h);
        for i in 0..w + 2 {
            v.push((i, 0));
        }
        for i in 0..w + 2 {
            v.push((i, h + 1));
        }
        for j in 1..=h {
            v.push((0, j));
            v.push((w + 1, j));
        }
        v
    }
    /// Padded ROI-layout coordinates of the inner 1-cell band.
    pub fn band(&self) -> Vec<(usize, usize)> {
        let (w, h) = (self.w, self.h);
        let mut v = vec![];
        for j in 1..=h {
            for i in 1..=w {
                if i == 1 || j == 1 || i == w || j == h {
                    v.push((i, j));
                }
            }
        }
        v
    }
    /// Full-layout padded coordinates of ROI-layout padded coordinates.
    pub fn to_full(&self, (i, j): (usize, usize)) -> (usize, usize) {
        (i + self.c0, j + self.r0)
    }
    pub fn contains_cell(&self, cell: u32, nx: usize) -> bool {
        let (c, r) = (cell as usize % nx, cell as usize / nx);
        c >= self.c0 && c < self.c0 + self.w && r >= self.r0 && r < self.r0 + self.h
    }
    /// Map a full-grid interior cell index to the ROI grid.
    pub fn map_cell(&self, cell: u32, nx: usize) -> u32 {
        let (c, r) = (cell as usize % nx, cell as usize / nx);
        ((r - self.r0) * self.w + (c - self.c0)) as u32
    }
}

/// Recorded ring and band states.
pub struct Tape<T: Real> {
    pub roi: Roi,
    ring_full: Vec<usize>,
    band_full: Vec<usize>,
    ring_sub: Vec<usize>,
    band_sub: Vec<usize>,
    /// 3 values (h, qx, qy) per ring cell per step.
    pub ring: Vec<T>,
    /// 3 values per band cell per sync point.
    pub band: Vec<T>,
    pub budget_bytes: usize,
    pub overflow: bool,
    pub sub_layout: Layout,
}

impl<T: Real> Tape<T> {
    pub fn new(roi: Roi, full: &Layout, budget_bytes: usize) -> Self {
        let sub = Layout::new::<T>(roi.w, roi.h);
        let ring = roi.ring();
        let band = roi.band();
        let fa = |v: &[(usize, usize)]| v.iter().map(|&p| roi.to_full(p)).map(|(i, j)| full.at(i, j)).collect();
        let sa = |v: &[(usize, usize)]| v.iter().map(|&(i, j)| sub.at(i, j)).collect();
        Self {
            roi,
            ring_full: fa(&ring),
            band_full: fa(&band),
            ring_sub: sa(&ring),
            band_sub: sa(&band),
            ring: vec![],
            band: vec![],
            budget_bytes,
            overflow: false,
            sub_layout: sub,
        }
    }
    pub fn bytes(&self) -> usize {
        (self.ring.capacity() + self.band.capacity()) * core::mem::size_of::<T>()
    }
    pub fn steps(&self) -> usize {
        self.ring.len() / (3 * self.ring_full.len())
    }
    /// Record the ring at the start of a step (after ghost fill), from full arrays.
    pub fn record_ring(&mut self, h: &[T], qx: &[T], qy: &[T]) {
        if self.overflow {
            return;
        }
        if (self.ring.len() + 3 * self.ring_full.len()) * core::mem::size_of::<T>() > self.budget_bytes {
            self.overflow = true;
            self.ring = vec![];
            self.band = vec![];
            return;
        }
        for &k in &self.ring_full {
            self.ring.extend_from_slice(&[h[k], qx[k], qy[k]]);
        }
    }
    pub fn record_band(&mut self, h: &[T], qx: &[T], qy: &[T]) {
        if self.overflow {
            return;
        }
        for &k in &self.band_full {
            self.band.extend_from_slice(&[h[k], qx[k], qy[k]]);
        }
    }
    /// Write the recorded ring of `step` into ROI-layout ghost cells; set ghost-wet flags.
    pub fn fill(&self, step: usize, h: &mut [T], qx: &mut [T], qy: &mut [T], ghost_wet: &mut [bool]) -> bool {
        let n = self.ring_sub.len();
        let off = step * 3 * n;
        if off + 3 * n > self.ring.len() {
            return false;
        }
        ghost_wet.iter_mut().for_each(|v| *v = false);
        let w = self.roi.w;
        for (k, &idx) in self.ring_sub.iter().enumerate() {
            let s = &self.ring[off + 3 * k..off + 3 * k + 3];
            h[idx] = s[0];
            qx[idx] = s[1];
            qy[idx] = s[2];
            if s[0] > T::ZERO {
                // Ring order: top row, bottom row, then (left, right) per interior row.
                let j = if k < w + 2 { 0 } else if k < 2 * (w + 2) { self.roi.h + 1 } else { 1 + (k - 2 * (w + 2)) / 2 };
                ghost_wet[j] = true;
            }
        }
        true
    }
    /// Max |Δh| and |Δq| between the ROI band now and the recorded band at sync `s` (1-based).
    pub fn band_diff(&self, sync: u32, h: &[T], qx: &[T], qy: &[T]) -> Option<(f64, f64)> {
        let n = self.band_sub.len();
        let off = (sync as usize - 1) * 3 * n;
        if off + 3 * n > self.band.len() {
            return None;
        }
        let (mut dh, mut dq) = (0.0f64, 0.0f64);
        for (k, &idx) in self.band_sub.iter().enumerate() {
            let s = &self.band[off + 3 * k..off + 3 * k + 3];
            dh = dh.max((h[idx].to_f64() - s[0].to_f64()).abs());
            dq = dq.max((qx[idx].to_f64() - s[1].to_f64()).abs()).max((qy[idx].to_f64() - s[2].to_f64()).abs());
        }
        Some((dh, dq))
    }
}

/// ROI terrain for one candidate: copies cell data and *all* face jumps (including the
/// faces between the ROI and its ring) from the candidate's full prepared terrain.
pub fn sub_terrain<T: Real>(terr: &PreparedTerrain<T>, roi: Roi) -> PreparedTerrain<T> {
    let b = &terr.base;
    let sl = Layout::new::<T>(roi.w, roi.h);
    let len = sl.len();
    let mut f = [vec![T::ZERO; len], vec![T::ZERO; len], vec![T::ZERO; len], vec![T::ZERO; len]];
    let (mut wet, mut gn2) = (vec![T::ZERO; len], vec![T::ZERO; len]);
    for js in 0..roi.h + 2 {
        let jf = js + roi.r0;
        for is in 0..roi.w + 2 {
            let i_f = is + roi.c0;
            let (ks, kf) = (sl.at(is, js), i_f);
            for (d, rows) in [(0usize, &terr.fx[0]), (1, &terr.fx[1]), (2, &terr.fy[0]), (3, &terr.fy[1])] {
                f[d][ks] = rows.row(jf)[kf];
            }
            wet[ks] = b.wet.row(jf)[kf];
            gn2[ks] = b.gn2.row(jf)[kf];
        }
    }
    let z_full = terr.z_level();
    let mut z = vec![0.0; roi.w * roi.h];
    let mut wall = Mask::new(roi.w, roi.h, false);
    let mut wet_count = 0;
    for r in 0..roi.h {
        for c in 0..roi.w {
            let k = (r + roi.r0) * b.nx() + c + roi.c0;
            z[r * roi.w + c] = z_full[k];
            let w = b.wall.data[k];
            wall.set(c, r, w);
            wet_count += !w as usize;
        }
    }
    let geo = GeoTransform {
        origin_x: b.geo.origin_x + roi.c0 as f64 * b.dx,
        origin_y: b.geo.origin_y - roi.r0 as f64 * b.dy,
        dx: b.dx,
        dy: b.dy,
    };
    let [f0, f1, f2, f3] = f;
    let base = TerrainBase {
        layout: sl,
        level: b.level,
        geo,
        dx: b.dx,
        dy: b.dy,
        mode: b.mode,
        fine: b.fine.clone(),
        z,
        wall,
        fx: [Arc::new(RowTable::from_full(sl, &f0, false)), Arc::new(RowTable::from_full(sl, &f1, false))],
        fy: [Arc::new(RowTable::from_full(sl, &f2, false)), Arc::new(RowTable::from_full(sl, &f3, false))],
        wet: RowTable::from_full(sl, &wet, true),
        gn2: RowTable::from_full(sl, &gn2, true),
        wet_count,
        boundaries: Boundaries::walls(roi.w, roi.h),
        g: b.g,
    };
    let base = Arc::new(base);
    PreparedTerrain {
        fx: [FaceRows::new(base.fx[0].clone()), FaceRows::new(base.fx[1].clone())],
        fy: [FaceRows::new(base.fy[0].clone()), FaceRows::new(base.fy[1].clone())],
        base,
        edit_rows: None,
        z_overrides: vec![],
    }
}

/// Extract the ROI part of a full checkpoint's state (ledger is not carried over).
pub fn sub_state<T: Real>(ck: &Checkpoint<T>, full: &Layout, roi: Roi, out_h: &mut [T], out_qx: &mut [T], out_qy: &mut [T]) {
    let sl = Layout::new::<T>(roi.w, roi.h);
    for js in 0..roi.h + 2 {
        for is in 0..roi.w + 2 {
            let (ks, kf) = (sl.at(is, js), full.at(is + roi.c0, js + roi.r0));
            out_h[ks] = ck.h[kf];
            out_qx[ks] = ck.qx[kf];
            out_qy[ks] = ck.qy[kf];
        }
    }
}
