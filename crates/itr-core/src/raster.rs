//! Plain (unpadded) row-major rasters and georeferencing.
//!
//! Convention (spec §4.1): row 0 is the **north** edge; `row` increases southward,
//! `col` increases eastward. World coordinates of the centre of cell `(col,row)` are
//! `x = origin_x + (col + 0.5)·dx`, `y = origin_y − (row + 0.5)·dy`, where
//! `(origin_x, origin_y)` is the north-west corner.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeoTransform {
    pub origin_x: f64,
    pub origin_y: f64,
    pub dx: f64,
    pub dy: f64,
}

impl GeoTransform {
    pub fn cell_center(&self, col: usize, row: usize) -> (f64, f64) {
        (
            self.origin_x + (col as f64 + 0.5) * self.dx,
            self.origin_y - (row as f64 + 0.5) * self.dy,
        )
    }
    /// Fractional (col,row) of a world point.
    pub fn to_grid(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.origin_x) / self.dx, (self.origin_y - y) / self.dy)
    }
    pub fn cell_area(&self) -> f64 {
        self.dx * self.dy
    }
}

#[derive(Clone, Debug)]
pub struct Raster {
    pub nx: usize,
    pub ny: usize,
    pub geo: GeoTransform,
    pub data: Vec<f64>,
}

impl Raster {
    pub fn new(nx: usize, ny: usize, geo: GeoTransform, fill: f64) -> Self {
        Self { nx, ny, geo, data: vec![fill; nx * ny] }
    }
    #[inline]
    pub fn idx(&self, col: usize, row: usize) -> usize {
        row * self.nx + col
    }
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> f64 {
        self.data[row * self.nx + col]
    }
    #[inline]
    pub fn set(&mut self, col: usize, row: usize, v: f64) {
        let i = self.idx(col, row);
        self.data[i] = v;
    }
    pub fn same_grid(&self, other: &Raster) -> bool {
        self.nx == other.nx && self.ny == other.ny && self.geo == other.geo
    }
}

/// Boolean mask on the same grid as a raster.
#[derive(Clone, Debug)]
pub struct Mask {
    pub nx: usize,
    pub ny: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(nx: usize, ny: usize, v: bool) -> Self {
        Self { nx, ny, data: vec![v; nx * ny] }
    }
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> bool {
        self.data[row * self.nx + col]
    }
    #[inline]
    pub fn set(&mut self, col: usize, row: usize, v: bool) {
        self.data[row * self.nx + col] = v;
    }
    pub fn count(&self) -> usize {
        self.data.iter().filter(|b| **b).count()
    }
    pub fn indices(&self) -> Vec<u32> {
        self.data.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect()
    }
    pub fn or(&mut self, other: &Mask) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a |= *b;
        }
    }
    /// 4-neighbour ring of cells just outside the mask (used for wall-mode buildings).
    pub fn outer_ring(&self) -> Mask {
        let mut out = Mask::new(self.nx, self.ny, false);
        for r in 0..self.ny {
            for c in 0..self.nx {
                if self.get(c, r) {
                    continue;
                }
                let nb = (c > 0 && self.get(c - 1, r))
                    || (c + 1 < self.nx && self.get(c + 1, r))
                    || (r > 0 && self.get(c, r - 1))
                    || (r + 1 < self.ny && self.get(c, r + 1));
                if nb {
                    out.set(c, r, true);
                }
            }
        }
        out
    }
}

/// Polygon with optional holes, in world coordinates of the raster's CRS.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polygon {
    pub exterior: Vec<(f64, f64)>,
    #[serde(default)]
    pub holes: Vec<Vec<(f64, f64)>>,
}

fn ring_contains(ring: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Polygon {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        ring_contains(&self.exterior, x, y) && !self.holes.iter().any(|h| ring_contains(h, x, y))
    }
    pub fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self { exterior: vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], holes: vec![] }
    }
}

/// Rasterize polygons with the **cell-centre rule** (spec §4.1): a cell belongs to a
/// polygon iff its centre lies inside it (even-odd, holes excluded).
pub fn rasterize(polys: &[Polygon], nx: usize, ny: usize, geo: &GeoTransform) -> Mask {
    let mut m = Mask::new(nx, ny, false);
    for p in polys {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in &p.exterior {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (c0, r1) = geo.to_grid(x0, y0);
        let (c1, r0) = geo.to_grid(x1, y1);
        let c0 = c0.floor().max(0.0) as usize;
        let r0 = r0.floor().max(0.0) as usize;
        let c1 = (c1.ceil() as isize).clamp(0, nx as isize) as usize;
        let r1 = (r1.ceil() as isize).clamp(0, ny as isize) as usize;
        for r in r0..r1 {
            for c in c0..c1 {
                let (x, y) = geo.cell_center(c, r);
                if p.contains(x, y) {
                    m.set(c, r, true);
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rect_rasterization_counts_cell_centres() {
        let geo = GeoTransform { origin_x: 0.0, origin_y: 10.0, dx: 1.0, dy: 1.0 };
        let m = rasterize(&[Polygon::rect(2.0, 2.0, 5.0, 4.0)], 10, 10, &geo);
        assert_eq!(m.count(), 6);
        // north-up: y in (2,4) → rows 6..8
        assert!(m.get(2, 6) && m.get(4, 7) && !m.get(5, 7) && !m.get(2, 5));
    }
}
