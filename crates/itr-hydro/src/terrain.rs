//! Prepared terrain (spec §7.3): per-face elevation jumps computed in f64 and rounded
//! once; the kernel never reads `z`. Row-level copy-on-write overlays let candidates
//! share the baseline terrain and patch only rows touched by an edit.

use crate::aligned::{AlignedBuf, Layout};
use itr_core::design::TerrainEdit;
use itr_core::raster::{GeoTransform, Mask, Raster};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_core::Real;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceMode {
    /// One signed jump per face (fine level).
    Signed,
    /// Two non-negative jumps to a face crest (coarse levels, §19.1).
    Crest,
}

/// Rows of a padded array, where many rows may share storage (uniform rows) and a
/// private overlay can replace individual rows (copy-on-write).
#[derive(Clone)]
pub struct RowTable<T: Real> {
    pub layout: Layout,
    pool: AlignedBuf<T>,
    /// Row j → row index in `pool`.
    map: Vec<u32>,
}

impl<T: Real> RowTable<T> {
    /// Build from a full padded array. With `dedupe`, bit-identical rows share storage
    /// (e.g. uniform roughness or wet-mask rows), so they stay hot in L1.
    pub fn from_full(layout: Layout, full: &[T], dedupe: bool) -> Self {
        let rows = layout.ny + 2;
        let mut map = vec![0u32; rows];
        let mut keep: Vec<usize> = vec![];
        let mut seen: std::collections::HashMap<Vec<u64>, u32> = std::collections::HashMap::new();
        for j in 0..rows {
            if dedupe {
                let key: Vec<u64> = full[layout.row(j)].iter().map(|v| v.to_f64().to_bits()).collect();
                if let Some(&s) = seen.get(&key) {
                    map[j] = s;
                    continue;
                }
                seen.insert(key, keep.len() as u32);
            }
            keep.push(j);
            map[j] = (keep.len() - 1) as u32;
        }
        let mut pool = AlignedBuf::new(keep.len().max(1) * layout.stride, T::ZERO);
        for (slot, &j) in keep.iter().enumerate() {
            let src = &full[j * layout.stride..(j + 1) * layout.stride];
            pool[slot * layout.stride..(slot + 1) * layout.stride].copy_from_slice(src);
        }
        Self { layout, pool, map }
    }
    #[inline(always)]
    pub fn row(&self, j: usize) -> &[T] {
        let s = self.map[j] as usize * self.layout.stride + self.layout.lp;
        &self.pool[s..s + self.layout.nx + 2]
    }
    pub fn distinct_rows(&self) -> usize {
        self.pool.len() / self.layout.stride
    }
}

/// Copy-on-write overlay over a shared `RowTable`.
pub struct FaceRows<T: Real> {
    base: Arc<RowTable<T>>,
    over_map: Vec<u32>,
    pool: Vec<T>,
    used: usize,
    touched: Vec<usize>,
}

const NONE: u32 = u32::MAX;

impl<T: Real> FaceRows<T> {
    pub fn new(base: Arc<RowTable<T>>) -> Self {
        let rows = base.layout.ny + 2;
        Self { base, over_map: vec![NONE; rows], pool: vec![], used: 0, touched: vec![] }
    }
    #[inline(always)]
    pub fn row(&self, j: usize) -> &[T] {
        let o = self.over_map[j];
        if o == NONE {
            self.base.row(j)
        } else {
            let n = self.base.layout.nx + 2;
            let s = o as usize * n;
            &self.pool[s..s + n]
        }
    }
    /// Private, mutable copy of row j (copied from base on first touch).
    pub fn row_mut(&mut self, j: usize) -> &mut [T] {
        let n = self.base.layout.nx + 2;
        if self.over_map[j] == NONE {
            if self.pool.len() < (self.used + 1) * n {
                self.pool.resize((self.used + 1) * n, T::ZERO);
            }
            let slot = self.used;
            self.used += 1;
            let src: Vec<T> = self.base.row(j).to_vec();
            self.pool[slot * n..(slot + 1) * n].copy_from_slice(&src);
            self.over_map[j] = slot as u32;
            self.touched.push(j);
        }
        let s = self.over_map[j] as usize * n;
        &mut self.pool[s..s + n]
    }
    /// Drop all private rows (O(touched rows), no copy).
    pub fn reset(&mut self) {
        for &j in &self.touched {
            self.over_map[j] = NONE;
        }
        self.touched.clear();
        self.used = 0;
    }
}

#[derive(Clone, Debug)]
pub enum Ghost {
    Wall,
    Transmissive,
    /// Water-surface elevation (re-based).
    Stage(f64),
    /// Inflow: hydrograph index and 1/segment-length (m⁻¹).
    Inflow(usize, f64),
}

/// Boundary specification per side cell (north/south: length nx; west/east: length ny).
#[derive(Clone, Debug)]
pub struct Boundaries {
    pub north: Vec<Ghost>,
    pub south: Vec<Ghost>,
    pub west: Vec<Ghost>,
    pub east: Vec<Ghost>,
    pub hydrographs: Vec<Hyetograph>,
}

impl Boundaries {
    pub fn walls(nx: usize, ny: usize) -> Self {
        Self {
            north: vec![Ghost::Wall; nx],
            south: vec![Ghost::Wall; nx],
            west: vec![Ghost::Wall; ny],
            east: vec![Ghost::Wall; ny],
            hydrographs: vec![],
        }
    }
    pub fn from_segments(segs: &[BoundarySegment], nx: usize, ny: usize, dx: f64, dy: f64, z_ref: f64) -> Self {
        let mut b = Self::walls(nx, ny);
        for s in segs {
            let (side, range) = match s {
                BoundarySegment::Stage { side, range_m, .. }
                | BoundarySegment::Inflow { side, range_m, .. }
                | BoundarySegment::Transmissive { side, range_m } => (*side, *range_m),
            };
            let (n, d) = match side {
                Side::North | Side::South => (nx, dx),
                Side::West | Side::East => (ny, dy),
            };
            let (a, e) = match range {
                Some([a, e]) => (((a / d).floor().max(0.0) as usize).min(n), ((e / d).ceil().max(0.0) as usize).min(n)),
                None => (0, n),
            };
            let len_m = (e.saturating_sub(a)) as f64 * d;
            let g = match s {
                BoundarySegment::Stage { stage_m, .. } => Ghost::Stage(stage_m - z_ref),
                BoundarySegment::Transmissive { .. } => Ghost::Transmissive,
                BoundarySegment::Inflow { hydrograph, .. } => {
                    b.hydrographs.push(Hyetograph::from_pairs(hydrograph));
                    Ghost::Inflow(b.hydrographs.len() - 1, if len_m > 0.0 { 1.0 / len_m } else { 0.0 })
                }
            };
            let v = match side {
                Side::North => &mut b.north,
                Side::South => &mut b.south,
                Side::West => &mut b.west,
                Side::East => &mut b.east,
            };
            for x in v.iter_mut().take(e).skip(a) {
                *x = g.clone();
            }
        }
        b
    }
}

/// Full-resolution terrain shared by all fidelity levels.
pub struct FineTerrain {
    pub nx: usize,
    pub ny: usize,
    pub geo: GeoTransform,
    /// Re-based elevations z − z_ref (walls already set to z_wall).
    pub z: Vec<f64>,
    pub wall: Mask,
    pub z_ref: f64,
    pub z_wall: f64,
    /// Manning n per cell.
    pub n: Vec<f64>,
}

impl FineTerrain {
    /// `dem`: absolute elevations; `nodata`/`obstacles` cells become walls.
    pub fn new(dem: &Raster, walls: &Mask, manning: &[f64]) -> Self {
        let mut zmin = f64::MAX;
        let mut zmax = f64::MIN;
        for (i, v) in dem.data.iter().enumerate() {
            if !walls.data[i] && v.is_finite() {
                zmin = zmin.min(*v);
                zmax = zmax.max(*v);
            }
        }
        if zmin > zmax {
            zmin = 0.0;
            zmax = 0.0;
        }
        let z_ref = zmin.floor();
        let z_wall = (zmax - z_ref) + 100.0;
        let z = dem
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| if walls.data[i] || !v.is_finite() { z_wall } else { v - z_ref })
            .collect();
        let mut wall = walls.clone();
        for (i, v) in dem.data.iter().enumerate() {
            if !v.is_finite() {
                wall.data[i] = true;
            }
        }
        Self { nx: dem.nx, ny: dem.ny, geo: dem.geo.clone(), z, wall, z_ref, z_wall, n: manning.to_vec() }
    }
}

/// Read-only, shareable prepared terrain for one fidelity level.
pub struct TerrainBase<T: Real> {
    pub layout: Layout,
    pub level: usize,
    pub geo: GeoTransform,
    pub dx: f64,
    pub dy: f64,
    pub mode: FaceMode,
    pub fine: Arc<FineTerrain>,
    /// Cell z at this level (re-based, unpadded nx*ny).
    pub z: Vec<f64>,
    pub wall: Mask,
    /// x-faces: signed dz in `fx[0]`; crest mode uses (`fx[0]`=d_l, `fx[1]`=d_r).
    pub fx: [Arc<RowTable<T>>; 2],
    pub fy: [Arc<RowTable<T>>; 2],
    pub wet: RowTable<T>,
    pub gn2: RowTable<T>,
    pub wet_count: usize,
    pub boundaries: Boundaries,
    pub g: f64,
}

pub const G: f64 = 9.81;

impl<T: Real> TerrainBase<T> {
    pub fn build(fine: Arc<FineTerrain>, level: usize, boundaries_segs: &[BoundarySegment]) -> Self {
        let lv = level.max(1);
        let nx = fine.nx.div_ceil(lv);
        let ny = fine.ny.div_ceil(lv);
        let geo = GeoTransform { origin_x: fine.geo.origin_x, origin_y: fine.geo.origin_y, dx: fine.geo.dx * lv as f64, dy: fine.geo.dy * lv as f64 };
        let layout = Layout::new::<T>(nx, ny);
        let mode = if lv == 1 { FaceMode::Signed } else { FaceMode::Crest };
        let (z, wall, nvals) = coarsen(&fine, lv, None);
        let len = layout.len();
        let mut fx0 = vec![T::ZERO; len];
        let mut fx1 = vec![T::ZERO; len];
        let mut fy0 = vec![T::ZERO; len];
        let mut fy1 = vec![T::ZERO; len];
        let zc = |c: usize, r: usize| z[r * nx + c];
        for j in 1..=ny {
            for i in 1..nx {
                let (a, b) = face_x(&fine, lv, mode, zc(i - 1, j - 1), zc(i, j - 1), i, j - 1, None);
                fx0[layout.at(i, j)] = T::from_f64(a);
                fx1[layout.at(i, j)] = T::from_f64(b);
            }
        }
        for j in 1..ny {
            for i in 1..=nx {
                let (a, b) = face_y(&fine, lv, mode, zc(i - 1, j - 1), zc(i - 1, j), i - 1, j, None);
                fy0[layout.at(i, j)] = T::from_f64(a);
                fy1[layout.at(i, j)] = T::from_f64(b);
            }
        }
        let mut wet = vec![T::ZERO; len];
        let mut gn2 = vec![T::ZERO; len];
        let mut wet_count = 0;
        for j in 1..=ny {
            for i in 1..=nx {
                let k = (j - 1) * nx + (i - 1);
                let w = !wall.data[k];
                wet[layout.at(i, j)] = if w { T::ONE } else { T::ZERO };
                wet_count += w as usize;
                gn2[layout.at(i, j)] = T::from_f64(G * nvals[k] * nvals[k]);
            }
        }
        let boundaries = Boundaries::from_segments(boundaries_segs, nx, ny, geo.dx, geo.dy, fine.z_ref);
        Self {
            layout,
            level: lv,
            dx: geo.dx,
            dy: geo.dy,
            geo,
            mode,
            fine: fine.clone(),
            z,
            wall,
            fx: [Arc::new(RowTable::from_full(layout, &fx0, false)), Arc::new(RowTable::from_full(layout, &fx1, true))],
            fy: [Arc::new(RowTable::from_full(layout, &fy0, false)), Arc::new(RowTable::from_full(layout, &fy1, true))],
            wet: RowTable::from_full(layout, &wet, true),
            gn2: RowTable::from_full(layout, &gn2, true),
            wet_count,
            boundaries,
            g: G,
        }
    }

    pub fn nx(&self) -> usize {
        self.layout.nx
    }
    pub fn ny(&self) -> usize {
        self.layout.ny
    }
    pub fn cell_area(&self) -> f64 {
        self.dx * self.dy
    }
}

/// Block-mean coarsening (walls excluded unless the block is all wall). With an edit,
/// only the edit's footprint is recomputed by the caller; here we coarsen everything.
fn coarsen(fine: &FineTerrain, lv: usize, edit: Option<&TerrainEdit>) -> (Vec<f64>, Mask, Vec<f64>) {
    let nx = fine.nx.div_ceil(lv);
    let ny = fine.ny.div_ceil(lv);
    let mut z = vec![0.0; nx * ny];
    let mut wall = Mask::new(nx, ny, false);
    let mut n = vec![0.0; nx * ny];
    for r in 0..ny {
        for c in 0..nx {
            let (zz, w, nn) = coarse_cell(fine, lv, c, r, edit);
            z[r * nx + c] = zz;
            wall.set(c, r, w);
            n[r * nx + c] = nn;
        }
    }
    (z, wall, n)
}

fn fine_z(fine: &FineTerrain, c: usize, r: usize, edit: Option<&TerrainEdit>) -> f64 {
    fine.z[r * fine.nx + c] + edit.map_or(0.0, |e| e.get(c, r))
}

fn coarse_cell(fine: &FineTerrain, lv: usize, c: usize, r: usize, edit: Option<&TerrainEdit>) -> (f64, bool, f64) {
    let (mut s, mut cnt, mut sn) = (0.0, 0usize, 0.0);
    let mut sn_all = 0.0;
    let mut tot = 0usize;
    for rr in r * lv..((r + 1) * lv).min(fine.ny) {
        for cc in c * lv..((c + 1) * lv).min(fine.nx) {
            tot += 1;
            sn_all += fine.n[rr * fine.nx + cc];
            if !fine.wall.get(cc, rr) {
                s += fine_z(fine, cc, rr, edit);
                sn += fine.n[rr * fine.nx + cc];
                cnt += 1;
            }
        }
    }
    if cnt == 0 { (fine.z_wall, true, sn_all / tot as f64) } else { (s / cnt as f64, false, sn / cnt as f64) }
}

/// x-face between coarse cells (cl, r) and (cl+1, r) — padded face index i = cl+1 at row r+1.
/// Returns (a, b): signed dz (a) in Signed mode; (d_l, d_r) in Crest mode.
#[allow(clippy::too_many_arguments)]
fn face_x(fine: &FineTerrain, lv: usize, mode: FaceMode, zl: f64, zr: f64, i: usize, r: usize, edit: Option<&TerrainEdit>) -> (f64, f64) {
    match mode {
        FaceMode::Signed => (zr - zl, 0.0),
        FaceMode::Crest => {
            // Face line between fine columns i*lv-1 and i*lv over the block's rows.
            let cl = i * lv - 1;
            let cr = (i * lv).min(fine.nx - 1);
            let mut crest = f64::MIN;
            let mut all_wall = true;
            for rr in r * lv..((r + 1) * lv).min(fine.ny) {
                for cc in [cl, cr] {
                    if !fine.wall.get(cc, rr) {
                        all_wall = false;
                        crest = crest.max(fine_z(fine, cc, rr, edit));
                    }
                }
            }
            if all_wall {
                crest = fine.z_wall;
            }
            let zf = crest.max(zl).max(zr);
            (zf - zl, zf - zr)
        }
    }
}

/// y-face between coarse cells (c, rt) [north] and (c, rt+1) [south]: padded index (c+1, rt+1).
#[allow(clippy::too_many_arguments)]
fn face_y(fine: &FineTerrain, lv: usize, mode: FaceMode, zt: f64, zb: f64, c: usize, rb: usize, edit: Option<&TerrainEdit>) -> (f64, f64) {
    match mode {
        FaceMode::Signed => (zb - zt, 0.0),
        FaceMode::Crest => {
            let rt_f = rb * lv - 1;
            let rb_f = (rb * lv).min(fine.ny - 1);
            let mut crest = f64::MIN;
            let mut all_wall = true;
            for cc in c * lv..((c + 1) * lv).min(fine.nx) {
                for rr in [rt_f, rb_f] {
                    if !fine.wall.get(cc, rr) {
                        all_wall = false;
                        crest = crest.max(fine_z(fine, cc, rr, edit));
                    }
                }
            }
            if all_wall {
                crest = fine.z_wall;
            }
            let zf = crest.max(zt).max(zb);
            (zf - zt, zf - zb)
        }
    }
}

/// Per-candidate prepared terrain: shared base + private copy-on-write face rows.
pub struct PreparedTerrain<T: Real> {
    pub base: Arc<TerrainBase<T>>,
    pub fx: [FaceRows<T>; 2],
    pub fy: [FaceRows<T>; 2],
    /// Padded rows touched by the current edit (for warm start), inclusive range.
    pub edit_rows: Option<(usize, usize)>,
    /// Edited cell z at this level for touched cells (sparse; for exports).
    pub z_overrides: Vec<(usize, f64)>,
}

impl<T: Real> PreparedTerrain<T> {
    pub fn new(base: Arc<TerrainBase<T>>) -> Self {
        let fx = [FaceRows::new(base.fx[0].clone()), FaceRows::new(base.fx[1].clone())];
        let fy = [FaceRows::new(base.fy[0].clone()), FaceRows::new(base.fy[1].clone())];
        Self { base, fx, fy, edit_rows: None, z_overrides: vec![] }
    }

    /// Restore baseline (O(touched rows)).
    pub fn reset(&mut self) {
        for f in self.fx.iter_mut().chain(self.fy.iter_mut()) {
            f.reset();
        }
        self.edit_rows = None;
        self.z_overrides.clear();
    }

    /// Apply a **fine-resolution** edit: recompute affected cells' z and adjacent face
    /// jumps at this level in O(bbox).
    pub fn apply_edit(&mut self, edit: &TerrainEdit) {
        self.reset();
        if edit.is_empty() {
            return;
        }
        let b = self.base.clone();
        let lv = b.level;
        let (nx, ny) = (b.nx(), b.ny());
        // Affected coarse cells.
        let c0 = edit.c0 / lv;
        let r0 = edit.r0 / lv;
        let c1 = ((edit.c0 + edit.w).div_ceil(lv)).min(nx);
        let r1 = ((edit.r0 + edit.h).div_ceil(lv)).min(ny);
        let mut zloc = std::collections::HashMap::new();
        for r in r0..r1 {
            for c in c0..c1 {
                let (z, _, _) = coarse_cell(&b.fine, lv, c, r, Some(edit));
                zloc.insert((c, r), z);
                self.z_overrides.push((r * nx + c, z));
            }
        }
        let zc = |c: usize, r: usize| *zloc.get(&(c, r)).unwrap_or(&b.z[r * nx + c]);
        // x-faces in rows r0..r1, face columns c0..=c1 (padded i = cl+1).
        for r in r0..r1 {
            for cl in c0.saturating_sub(1)..c1.min(nx - 1) {
                let i = cl + 1;
                let (a, bb) = face_x(&b.fine, lv, b.mode, zc(cl, r), zc(cl + 1, r), i, r, Some(edit));
                self.fx[0].row_mut(r + 1)[i] = T::from_f64(a);
                if b.mode == FaceMode::Crest {
                    self.fx[1].row_mut(r + 1)[i] = T::from_f64(bb);
                }
            }
        }
        // y-faces between rows rt and rt+1 for rt in r0-1..r1, columns c0..c1.
        for rt in r0.saturating_sub(1)..r1.min(ny - 1) {
            for c in c0..c1 {
                let (a, bb) = face_y(&b.fine, lv, b.mode, zc(c, rt), zc(c, rt + 1), c, rt + 1, Some(edit));
                self.fy[0].row_mut(rt + 1)[c + 1] = T::from_f64(a);
                if b.mode == FaceMode::Crest {
                    self.fy[1].row_mut(rt + 1)[c + 1] = T::from_f64(bb);
                }
            }
        }
        // Padded rows whose fluxes can change: r0..r1 (+1 padding) ± 1.
        self.edit_rows = Some((r0, (r1 + 1).min(ny + 1)));
    }

    /// Cell z at this level including the current edit (re-based).
    pub fn z_level(&self) -> Vec<f64> {
        let mut z = self.base.z.clone();
        for &(k, v) in &self.z_overrides {
            z[k] = v;
        }
        z
    }
}
