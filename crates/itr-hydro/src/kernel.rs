//! Step kernels (spec §7.4).
//!
//! * [`step_oracle`]: straightforward reference. Velocities and all face fluxes are
//!   stored in full arrays, then cells are updated. Never skips work.
//! * [`step_fused`]: single in-place pass with row buffers, parallel strips, dry-row
//!   skipping and fused metrics.
//!
//! Both call the same row functions (structure-of-arrays, branch-free, two-pass row
//! update), so their results are bitwise identical (tested).

use crate::aligned::Layout;
use crate::numerics::{cell_vel, face_crest, face_signed, flux_update, sources, Face, StepConsts, Vel};
use crate::terrain::{FaceMode, PreparedTerrain};
use itr_core::{max_slice, FixedSum, Real};
use crate::pool::{GridPool, Shards};

/// Structure-of-arrays row of cell velocities.
#[derive(Clone, Default)]
pub struct VelRow<T: Real> {
    pub u: Vec<T>,
    pub v: Vec<T>,
    pub c: Vec<T>,
}

impl<T: Real> VelRow<T> {
    pub fn new(n: usize) -> Self {
        Self { u: vec![T::ZERO; n], v: vec![T::ZERO; n], c: vec![T::ZERO; n] }
    }
}

/// Structure-of-arrays row of face fluxes.
#[derive(Clone, Default)]
pub struct FaceRow<T: Real> {
    pub m: Vec<T>,
    pub nl: Vec<T>,
    pub nr: Vec<T>,
    pub t: Vec<T>,
}

impl<T: Real> FaceRow<T> {
    pub fn new(n: usize) -> Self {
        Self { m: vec![T::ZERO; n], nl: vec![T::ZERO; n], nr: vec![T::ZERO; n], t: vec![T::ZERO; n] }
    }
    fn zero(&mut self) {
        for a in [&mut self.m, &mut self.nl, &mut self.nr, &mut self.t] {
            a.fill(T::ZERO);
        }
    }
    fn copy_from(&mut self, o: &FaceRow<T>) {
        self.m.copy_from_slice(&o.m);
        self.nl.copy_from_slice(&o.nl);
        self.nr.copy_from_slice(&o.nr);
        self.t.copy_from_slice(&o.t);
    }
}

/// Per-row scratch for the two-pass update (sums/max are taken in pass 2).
#[derive(Clone)]
pub struct RowScratch<T: Real> {
    infil: Vec<T>,
    fix: Vec<T>,
    s: Vec<T>,
}

impl<T: Real> RowScratch<T> {
    pub fn new(nx: usize) -> Self {
        let n = (nx + 2).div_ceil(16) * 16;
        Self { infil: vec![T::ZERO; n], fix: vec![T::ZERO; n], s: vec![T::ZERO; n] }
    }
}

/// Per-row accounting produced by the update of one row.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowAcc {
    pub infil: f64,
    pub fix: f64,
    pub bnd_in: f64,
    pub bnd_out: f64,
    pub smax: f64,
    pub sum_s: f64,
    pub wet: bool,
    pub bad: bool,
    pub skipped: bool,
}

/// Monitored-cell running maxima (spec §7.4).
pub struct Monitors {
    pub nx: usize,
    pub cells: Vec<u32>,
    /// For interior row r: monitors cells[row_start[r]..row_start[r+1]].
    pub row_start: Vec<usize>,
    pub max: Vec<f64>,
    pub tmax: Vec<f64>,
}

impl Monitors {
    pub fn new(cells: &[u32], nx: usize, ny: usize) -> Self {
        let mut row_start = vec![0usize; ny + 1];
        let mut k = 0;
        for (r, rs) in row_start.iter_mut().enumerate().take(ny) {
            *rs = k;
            while k < cells.len() && (cells[k] as usize) / nx == r {
                k += 1;
            }
        }
        row_start[ny] = cells.len();
        Self { nx, cells: cells.to_vec(), row_start, max: vec![0.0; cells.len()], tmax: vec![0.0; cells.len()] }
    }
    pub fn reset(&mut self) {
        self.max.iter_mut().for_each(|v| *v = 0.0);
        self.tmax.iter_mut().for_each(|v| *v = 0.0);
    }
}

#[inline(never)]
fn vel_row<T: Real>(h: &[T], qx: &[T], qy: &[T], out: &mut VelRow<T>, k: &StepConsts<T>) {
    let n = out.u.len();
    let (h, qx, qy) = (&h[..n], &qx[..n], &qy[..n]);
    let (u, v, c) = (&mut out.u[..n], &mut out.v[..n], &mut out.c[..n]);
    for i in 0..n {
        let o = cell_vel(h[i], qx[i], qy[i], k);
        u[i] = o.u;
        v[i] = o.v;
        c[i] = o.c;
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn xface_row<T: Real, const CREST: bool>(h: &[T], v: &VelRow<T>, f0: &[T], f1: &[T], nx: usize, out: &mut FaceRow<T>, k: &StepConsts<T>) {
    // Every slice is an offset window of exactly `m` elements and the loop runs 0..m, so
    // LLVM elides all bounds checks and vectorizes (spec §7.4).
    let m = nx + 1;
    let (hl, hr) = (&h[0..m], &h[1..m + 1]);
    let (ul, ur) = (&v.u[0..m], &v.u[1..m + 1]);
    let (vl, vr) = (&v.v[0..m], &v.v[1..m + 1]);
    let (cl, cr) = (&v.c[0..m], &v.c[1..m + 1]);
    let (f0, f1) = (&f0[0..m], &f1[0..m]);
    let (om, onl, onr, ot) = (&mut out.m[0..m], &mut out.nl[0..m], &mut out.nr[0..m], &mut out.t[0..m]);
    for i in 0..m {
        let a = Vel { u: ul[i], v: vl[i], c: cl[i] };
        let b = Vel { u: ur[i], v: vr[i], c: cr[i] };
        let f = if CREST { face_crest(hl[i], a, hr[i], b, f0[i], f1[i], true, k) } else { face_signed(hl[i], a, hr[i], b, f0[i], true, k) };
        om[i] = f.m;
        onl[i] = f.n_l;
        onr[i] = f.n_r;
        ot[i] = f.t;
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn yface_row<T: Real, const CREST: bool>(
    ht: &[T],
    vt: &VelRow<T>,
    hb: &[T],
    vb: &VelRow<T>,
    f0: &[T],
    f1: &[T],
    nx: usize,
    out: &mut FaceRow<T>,
    k: &StepConsts<T>,
) {
    let m = nx;
    let r = 1..m + 1;
    let (ht, hb, f0, f1) = (&ht[r.clone()], &hb[r.clone()], &f0[r.clone()], &f1[r.clone()]);
    let (tu, tv, tc) = (&vt.u[r.clone()], &vt.v[r.clone()], &vt.c[r.clone()]);
    let (bu, bv, bc) = (&vb.u[r.clone()], &vb.v[r.clone()], &vb.c[r.clone()]);
    let (om, onl, onr, ot) = (&mut out.m[r.clone()], &mut out.nl[r.clone()], &mut out.nr[r.clone()], &mut out.t[r]);
    for i in 0..m {
        let a = Vel { u: tu[i], v: tv[i], c: tc[i] };
        let b = Vel { u: bu[i], v: bv[i], c: bc[i] };
        let f = if CREST { face_crest(ht[i], a, hb[i], b, f0[i], f1[i], false, k) } else { face_signed(ht[i], a, hb[i], b, f0[i], false, k) };
        om[i] = f.m;
        onl[i] = f.n_l;
        onr[i] = f.n_r;
        ot[i] = f.t;
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn update_cells<T: Real>(
    hc: &mut [T],
    qxc: &mut [T],
    qyc: &mut [T],
    wm: &[T],
    wnr: &[T],
    wt: &[T],
    em: &[T],
    enl: &[T],
    et: &[T],
    nm: &[T],
    nnr: &[T],
    nt: &[T],
    sm: &[T],
    snl: &[T],
    st: &[T],
    wet: &[T],
    gn2: &[T],
    infil: &mut [T],
    fix: &mut [T],
    srate: &mut [T],
    k: &StepConsts<T>,
) -> bool {
    let m = hc.len();
    let (qxc, qyc, wet, gn2, infil, fix, srate) = (&mut qxc[..m], &mut qyc[..m], &wet[..m], &gn2[..m], &mut infil[..m], &mut fix[..m], &mut srate[..m]);
    let (wm, wnr, wt, em, enl, et) = (&wm[..m], &wnr[..m], &wt[..m], &em[..m], &enl[..m], &et[..m]);
    let (nm, nnr, nt, sm, snl, st) = (&nm[..m], &nnr[..m], &nt[..m], &sm[..m], &snl[..m], &st[..m]);
    let z = T::ZERO;
    // Loop 1: flux divergence (15 input streams). Loop 2: sources and friction (5 input
    // streams). Split so each loop stays under LLVM's vectorization limits; values pass
    // through memory at working precision, so results equal the fused `update_cell`.
    for i in 0..m {
        let fw = Face { m: wm[i], n_l: z, n_r: wnr[i], t: wt[i] };
        let fe = Face { m: em[i], n_l: enl[i], n_r: z, t: et[i] };
        let fnn = Face { m: nm[i], n_l: z, n_r: nnr[i], t: nt[i] };
        let fs = Face { m: sm[i], n_l: snl[i], n_r: z, t: st[i] };
        let (a, b, c) = flux_update(hc[i], qxc[i], qyc[i], &fw, &fe, &fnn, &fs, k);
        hc[i] = a;
        qxc[i] = b;
        qyc[i] = c;
    }
    let mut bad = false;
    for i in 0..m {
        let o = sources(hc[i], qxc[i], qyc[i], wet[i], gn2[i], k);
        hc[i] = o.h;
        qxc[i] = o.qx;
        qyc[i] = o.qy;
        infil[i] = o.infil;
        fix[i] = o.fix;
        srate[i] = o.s;
        bad |= o.bad;
    }
    bad
}

/// Update one interior row in place from its face rows. Shared by both kernels.
/// Pass 1: per-cell arithmetic. Pass 2: fixed-layout sums and maxima.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn update_row<T: Real>(
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    fx: &FaceRow<T>,
    gn: &FaceRow<T>,
    gs: &FaceRow<T>,
    wet: &[T],
    gn2: &[T],
    nx: usize,
    k: &StepConsts<T>,
    rs: &mut RowScratch<T>,
    want_sum_s: bool,
) -> (RowAcc, f64) {
    let m = nx;
    let c = 1..m + 1; // interior cells
    // West face of cell i is x-face i−1, east face is x-face i. Every array is passed to
    // `update_cells` as its own slice argument, so each is `noalias` and LLVM vectorizes
    // without runtime overlap checks.
    let bad = update_cells(
        &mut h[c.clone()],
        &mut qx[c.clone()],
        &mut qy[c.clone()],
        &fx.m[0..m],
        &fx.nr[0..m],
        &fx.t[0..m],
        &fx.m[1..m + 1],
        &fx.nl[1..m + 1],
        &fx.t[1..m + 1],
        &gn.m[c.clone()],
        &gn.nr[c.clone()],
        &gn.t[c.clone()],
        &gs.m[c.clone()],
        &gs.nl[c.clone()],
        &gs.t[c.clone()],
        &wet[c.clone()],
        &gn2[c.clone()],
        &mut rs.infil[c.clone()],
        &mut rs.fix[c.clone()],
        &mut rs.s[c],
        k,
    );
    let n = nx + 2;
    let (infil, fix, s) = (&rs.infil[..n], &rs.fix[..n], &rs.s[..n]);
    let mut si = FixedSum::<T>::default();
    si.add_slice(infil);
    let mut sf = FixedSum::<T>::default();
    sf.add_slice(fix);
    let sum_s = if want_sum_s {
        let mut ss = FixedSum::<T>::default();
        ss.add_slice(s);
        ss.total().to_f64()
    } else {
        0.0
    };
    let wet_any = max_slice(&h[1..n - 1]) > T::ZERO;
    (
        RowAcc {
            infil: si.total().to_f64(),
            fix: sf.total().to_f64(),
            smax: max_slice(s).to_f64(),
            wet: wet_any,
            bad,
            ..Default::default()
        },
        sum_s,
    )
}

#[inline(always)]
fn monitors_row<T: Real>(mon_cells: &[u32], mon_max: &mut [f64], mon_tmax: &mut [f64], h: &[T], nx: usize, t_new: f64) {
    for (k, &c) in mon_cells.iter().enumerate() {
        let col = c as usize % nx;
        let v = h[col + 1].to_f64();
        if v > mon_max[k] {
            mon_max[k] = v;
            mon_tmax[k] = t_new;
        }
    }
}

/// Outputs of one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub smax: f64,
    pub sum_s: f64,
    pub infil: f64,
    pub fix: f64,
    pub bnd_in: f64,
    pub bnd_out: f64,
    pub bad: bool,
    pub rows_skipped: u64,
}

/// Per-strip scratch buffers (row buffers, spec §7.4).
pub struct StripBuf<T: Real> {
    vel_cur: VelRow<T>,
    vel_next: VelRow<T>,
    fx: FaceRow<T>,
    g_prev: FaceRow<T>,
    g_next: FaceRow<T>,
    rs: RowScratch<T>,
    acc: Vec<RowAcc>,
    /// Strip-local monitor maxima (rows of this strip), so strips never share writes.
    mon_off: usize,
    mon_max: Vec<f64>,
    mon_tmax: Vec<f64>,
}

impl<T: Real> StripBuf<T> {
    pub fn new(nx: usize, rows: usize) -> Self {
        Self {
            vel_cur: VelRow::new(nx + 2),
            vel_next: VelRow::new(nx + 2),
            fx: FaceRow::new(nx + 2),
            g_prev: FaceRow::new(nx + 2),
            g_next: FaceRow::new(nx + 2),
            rs: RowScratch::new(nx),
            acc: vec![RowAcc::default(); rows],
            mon_off: 0,
            mon_max: vec![],
            mon_tmax: vec![],
        }
    }
}

/// Kernel scratch owned by the workspace.
pub struct KernelScratch<T: Real> {
    pub strip_rows: usize,
    pub strips: Vec<StripBuf<T>>,
    /// Boundary y-face rows at strip starts (face between rows a−1 and a) and at the
    /// bottom (ny, ny+1). `bnd_rows[s]` = top face of strip s; last = bottom face.
    pub bnd_rows: Vec<FaceRow<T>>,
    pub tmp_vel: [VelRow<T>; 2],
    /// Oracle-only full arrays.
    pub oracle_vel: Vec<VelRow<T>>,
    pub oracle_fx: Vec<FaceRow<T>>,
    pub oracle_fy: Vec<FaceRow<T>>,
    pub oracle_rs: RowScratch<T>,
    /// Dryness of padded rows at the start of the step (interior part).
    pub row_dry: Vec<bool>,
    pub ghost_wet: Vec<bool>,
    new_dry: Vec<bool>,
    sum_s_rows: Vec<f64>,
}

impl<T: Real> KernelScratch<T> {
    pub fn new(l: &Layout, strip_rows: usize) -> Self {
        let (nx, ny) = (l.nx, l.ny);
        let s = strip_rows.max(1).min(ny.max(1));
        let n_strips = ny.div_ceil(s);
        Self {
            strip_rows: s,
            strips: (0..n_strips).map(|k| StripBuf::new(nx, s.min(ny - k * s))).collect(),
            bnd_rows: (0..=n_strips).map(|_| FaceRow::new(nx + 2)).collect(),
            tmp_vel: [VelRow::new(nx + 2), VelRow::new(nx + 2)],
            oracle_vel: vec![],
            oracle_fx: vec![],
            oracle_fy: vec![],
            oracle_rs: RowScratch::new(nx),
            row_dry: vec![false; ny + 2],
            ghost_wet: vec![false; ny + 2],
            new_dry: vec![false; ny + 2],
            sum_s_rows: vec![0.0; ny],
        }
    }

    /// Distribute canonical monitor maxima into strip-local buffers.
    pub fn scatter_monitors(&mut self, mon: &Monitors) {
        let s_rows = self.strip_rows;
        let ny = mon.row_start.len() - 1;
        for (s, b) in self.strips.iter_mut().enumerate() {
            let a = s * s_rows;
            let e = ((s + 1) * s_rows).min(ny);
            let (lo, hi) = (mon.row_start[a], mon.row_start[e]);
            b.mon_off = lo;
            b.mon_max.clear();
            b.mon_max.extend_from_slice(&mon.max[lo..hi]);
            b.mon_tmax.clear();
            b.mon_tmax.extend_from_slice(&mon.tmax[lo..hi]);
        }
    }

    /// Copy strip-local monitor maxima back into the canonical arrays.
    pub fn gather_monitors(&self, mon: &mut Monitors) {
        for b in &self.strips {
            let n = b.mon_max.len();
            mon.max[b.mon_off..b.mon_off + n].copy_from_slice(&b.mon_max);
            mon.tmax[b.mon_off..b.mon_off + n].copy_from_slice(&b.mon_tmax);
        }
    }
}

/// Boundary-face mass flux accounting for x-faces of a row (west face 0, east face nx).
#[inline(always)]
fn x_boundary<T: Real>(fx: &FaceRow<T>, nx: usize, dt_dy: f64, acc: &mut RowAcc) {
    let w = fx.m[0].to_f64() * dt_dy;
    let e = fx.m[nx].to_f64() * dt_dy;
    if w > 0.0 {
        acc.bnd_in += w
    } else {
        acc.bnd_out -= w
    }
    if e > 0.0 {
        acc.bnd_out += e
    } else {
        acc.bnd_in -= e
    }
}

/// y-boundary (top: face 1/2, inflow when m>0; bottom: face ny+1/2, outflow when m>0).
fn y_boundary<T: Real>(row: &FaceRow<T>, nx: usize, dt_dx: f64, top: bool) -> (f64, f64) {
    let mut s = FixedSum::<f64>::default();
    for i in 1..=nx {
        s.add(i, row.m[i].to_f64());
    }
    let v = s.total() * dt_dx;
    let (mut fin, mut fout) = (0.0, 0.0);
    if top {
        if v > 0.0 {
            fin = v
        } else {
            fout = -v
        }
    } else if v > 0.0 {
        fout = v
    } else {
        fin = -v
    }
    (fin, fout)
}

pub struct StepCtx<'a, T: Real> {
    pub l: &'a Layout,
    pub terr: &'a PreparedTerrain<T>,
    pub k: StepConsts<T>,
    pub t_new: f64,
    pub dt: f64,
    pub want_sum_s: bool,
    pub skip_dry: bool,
}

/// Finalize per-row accounting into a step summary, in row order (deterministic).
fn finalize(accs: impl Iterator<Item = RowAcc>, top: (f64, f64), bot: (f64, f64), sum_s_rows: &[f64]) -> StepStats {
    let mut st = StepStats { bnd_in: top.0 + bot.0, bnd_out: top.1 + bot.1, ..Default::default() };
    for a in accs {
        st.infil += a.infil;
        st.fix += a.fix;
        st.bnd_in += a.bnd_in;
        st.bnd_out += a.bnd_out;
        st.smax = st.smax.max(a.smax);
        st.bad |= a.bad;
        st.rows_skipped += a.skipped as u64;
    }
    for s in sum_s_rows {
        st.sum_s += s;
    }
    st
}

/// Reference kernel (spec §7.2): full arrays, no fusion, no skipping.
pub fn step_oracle<T: Real>(cx: &StepCtx<T>, h: &mut [T], qx: &mut [T], qy: &mut [T], sc: &mut KernelScratch<T>, mon: &mut Monitors) -> StepStats {
    match cx.terr.base.mode {
        FaceMode::Signed => step_oracle_m::<T, false>(cx, h, qx, qy, sc, mon),
        FaceMode::Crest => step_oracle_m::<T, true>(cx, h, qx, qy, sc, mon),
    }
}

fn step_oracle_m<T: Real, const CREST: bool>(
    cx: &StepCtx<T>,
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    sc: &mut KernelScratch<T>,
    mon: &mut Monitors,
) -> StepStats {
    let l = cx.l;
    let (nx, ny) = (l.nx, l.ny);
    let k = &cx.k;
    if sc.oracle_vel.len() != ny + 2 {
        sc.oracle_vel = (0..ny + 2).map(|_| VelRow::new(nx + 2)).collect();
        sc.oracle_fx = (0..ny + 2).map(|_| FaceRow::new(nx + 2)).collect();
        sc.oracle_fy = (0..ny + 1).map(|_| FaceRow::new(nx + 2)).collect();
    }
    for j in 0..ny + 2 {
        let r = l.row(j);
        vel_row(&h[r.clone()], &qx[r.clone()], &qy[r], &mut sc.oracle_vel[j], k);
    }
    for j in 1..=ny {
        let r = l.row(j);
        xface_row::<T, CREST>(&h[r], &sc.oracle_vel[j], cx.terr.fx[0].row(j), cx.terr.fx[1].row(j), nx, &mut sc.oracle_fx[j], k);
    }
    for j in 0..=ny {
        let (rt, rb) = (l.row(j), l.row(j + 1));
        yface_row::<T, CREST>(&h[rt], &sc.oracle_vel[j], &h[rb], &sc.oracle_vel[j + 1], cx.terr.fy[0].row(j), cx.terr.fy[1].row(j), nx, &mut sc.oracle_fy[j], k);
    }
    let dt_dy = cx.dt * cx.terr.base.dy;
    let dt_dx = cx.dt * cx.terr.base.dx;
    let mut st_accs = Vec::with_capacity(ny);
    for j in 1..=ny {
        let r = l.row(j);
        let (h_r, qx_r, qy_r) = (&mut h[r.clone()], &mut qx[r.clone()], &mut qy[r]);
        let (mut a, ss) = update_row(
            h_r,
            qx_r,
            qy_r,
            &sc.oracle_fx[j],
            &sc.oracle_fy[j - 1],
            &sc.oracle_fy[j],
            cx.terr.base.wet.row(j),
            cx.terr.base.gn2.row(j),
            nx,
            k,
            &mut sc.oracle_rs,
            cx.want_sum_s,
        );
        x_boundary(&sc.oracle_fx[j], nx, dt_dy, &mut a);
        let (s0, s1) = (mon.row_start[j - 1], mon.row_start[j]);
        monitors_row(&mon.cells[s0..s1], &mut mon.max[s0..s1], &mut mon.tmax[s0..s1], h_r, nx, cx.t_new);
        sc.row_dry[j] = !a.wet;
        sc.sum_s_rows[j - 1] = ss;
        st_accs.push(a);
    }
    sc.row_dry[0] = true;
    sc.row_dry[ny + 1] = true;
    let top = y_boundary(&sc.oracle_fy[0], nx, dt_dx, true);
    let bot = y_boundary(&sc.oracle_fy[ny], nx, dt_dx, false);
    finalize(st_accs.into_iter(), top, bot, &sc.sum_s_rows)
}

/// Fused in-place kernel with parallel strips (spec §7.4).
pub fn step_fused<T: Real>(
    cx: &StepCtx<T>,
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    sc: &mut KernelScratch<T>,
    mon: &mut Monitors,
    pool: Option<&GridPool>,
) -> StepStats {
    match cx.terr.base.mode {
        FaceMode::Signed => step_fused_m::<T, false>(cx, h, qx, qy, sc, mon, pool),
        FaceMode::Crest => step_fused_m::<T, true>(cx, h, qx, qy, sc, mon, pool),
    }
}

type StripItem<'a, T> = (usize, (((((&'a mut [T], &'a mut [T]), &'a mut [T]), &'a mut StripBuf<T>), &'a mut [bool]), &'a mut [f64]));

#[allow(clippy::too_many_arguments)]
fn step_fused_m<T: Real, const CREST: bool>(
    cx: &StepCtx<T>,
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    sc: &mut KernelScratch<T>,
    mon: &mut Monitors,
    pool: Option<&GridPool>,
) -> StepStats {
    let l = *cx.l;
    let (nx, ny) = (l.nx, l.ny);
    let k = cx.k;
    let s_rows = sc.strip_rows;
    let n_strips = sc.strips.len();
    let rain_zero = !(k.rain > T::ZERO);

    // Row dryness at step start: interior dryness from the previous update + ghosts.
    let mut dry = std::mem::take(&mut sc.row_dry);
    for j in 0..ny + 2 {
        dry[j] = dry[j] && !sc.ghost_wet[j];
    }
    let skippable = |j: usize| cx.skip_dry && rain_zero && dry[j - 1] && dry[j] && dry[j + 1];

    // Pre-phase: boundary y-face rows at strip starts and bottom, from the old state.
    for s in 0..=n_strips {
        let a = if s < n_strips { 1 + s * s_rows } else { ny + 1 };
        let (rt, rb) = (l.row(a - 1), l.row(a));
        let [vt, vb] = &mut sc.tmp_vel;
        vel_row(&h[rt.clone()], &qx[rt.clone()], &qy[rt.clone()], vt, &k);
        vel_row(&h[rb.clone()], &qx[rb.clone()], &qy[rb.clone()], vb, &k);
        yface_row::<T, CREST>(&h[rt], vt, &h[rb], vb, cx.terr.fy[0].row(a - 1), cx.terr.fy[1].row(a - 1), nx, &mut sc.bnd_rows[s], &k);
    }
    let dt_dy = cx.dt * cx.terr.base.dy;
    let dt_dx = cx.dt * cx.terr.base.dx;
    let top = y_boundary(&sc.bnd_rows[0], nx, dt_dx, true);
    let bot = y_boundary(&sc.bnd_rows[n_strips], nx, dt_dx, false);

    // Split state into strips of whole rows (interior rows 1..=ny). No allocation:
    // strips are zipped chunk iterators over the state and preallocated scratch.
    let stride = l.stride;
    let interior = stride..(ny + 1) * stride;
    let chunk = s_rows * stride;
    let terr = cx.terr;
    let bnd = &sc.bnd_rows;
    let mon_cells = &mon.cells;
    let row_start = &mon.row_start;
    let work = |(s, (((((hs, qxs), qys), buf), nd), ssr)): StripItem<'_, T>| {
        let a = 1 + s * s_rows;
        let b = (a + s_rows).min(ny + 1);
        let row_rng = |j: usize| {
            let o = (j - a) * stride + l.lp;
            o..o + nx + 2
        };
        let mon_base = buf.mon_off;
        buf.g_prev.copy_from(&bnd[s]);
        let mut cur_valid = false;
        for j in a..b {
            let acc_slot = j - a;
            if skippable(j) {
                buf.acc[acc_slot] = RowAcc { skipped: true, ..Default::default() };
                nd[acc_slot] = true;
                ssr[acc_slot] = 0.0;
                buf.g_next.zero();
                std::mem::swap(&mut buf.g_prev, &mut buf.g_next);
                cur_valid = false;
                continue;
            }
            let rj = row_rng(j);
            if !cur_valid {
                vel_row(&hs[rj.clone()], &qxs[rj.clone()], &qys[rj.clone()], &mut buf.vel_cur, &k);
            }
            if j + 1 < b {
                let rn = row_rng(j + 1);
                vel_row(&hs[rn.clone()], &qxs[rn.clone()], &qys[rn.clone()], &mut buf.vel_next, &k);
                yface_row::<T, CREST>(&hs[rj.clone()], &buf.vel_cur, &hs[rn], &buf.vel_next, terr.fy[0].row(j), terr.fy[1].row(j), nx, &mut buf.g_next, &k);
                cur_valid = true;
            } else {
                buf.g_next.copy_from(&bnd[s + 1]);
                cur_valid = false;
            }
            xface_row::<T, CREST>(&hs[rj.clone()], &buf.vel_cur, terr.fx[0].row(j), terr.fx[1].row(j), nx, &mut buf.fx, &k);
            let (mut acc, ss) = update_row(
                &mut hs[rj.clone()],
                &mut qxs[rj.clone()],
                &mut qys[rj.clone()],
                &buf.fx,
                &buf.g_prev,
                &buf.g_next,
                terr.base.wet.row(j),
                terr.base.gn2.row(j),
                nx,
                &k,
                &mut buf.rs,
                cx.want_sum_s,
            );
            x_boundary(&buf.fx, nx, dt_dy, &mut acc);
            let (s0, s1) = (row_start[j - 1] - mon_base, row_start[j] - mon_base);
            monitors_row(&mon_cells[row_start[j - 1]..row_start[j]], &mut buf.mon_max[s0..s1], &mut buf.mon_tmax[s0..s1], &hs[rj], nx, cx.t_new);
            nd[acc_slot] = !acc.wet;
            ssr[acc_slot] = ss;
            buf.acc[acc_slot] = acc;
            std::mem::swap(&mut buf.g_prev, &mut buf.g_next);
            if cur_valid {
                std::mem::swap(&mut buf.vel_cur, &mut buf.vel_next);
            }
        }
    };
    {
        let new_dry = &mut sc.new_dry;
        let sum_s_rows = &mut sc.sum_s_rows;
        match pool {
            Some(p) if n_strips > 1 => {
                let (hs, qxs, qys) = (Shards::new(&mut h[interior.clone()]), Shards::new(&mut qx[interior.clone()]), Shards::new(&mut qy[interior]));
                let (bufs, nds, ssrs) = (Shards::new(&mut sc.strips), Shards::new(&mut new_dry[1..=ny]), Shards::new(sum_s_rows));
                // SAFETY: task s touches only chunk s of each buffer.
                p.run(n_strips, &|s| unsafe {
                    work((s, (((((hs.chunk(s, chunk), qxs.chunk(s, chunk)), qys.chunk(s, chunk)), &mut bufs.chunk(s, 1)[0]), nds.chunk(s, s_rows)), ssrs.chunk(s, s_rows))))
                });
            }
            _ => h[interior.clone()]
                .chunks_mut(chunk)
                .zip(qx[interior.clone()].chunks_mut(chunk))
                .zip(qy[interior].chunks_mut(chunk))
                .zip(sc.strips.iter_mut())
                .zip(new_dry[1..=ny].chunks_mut(s_rows))
                .zip(sum_s_rows.chunks_mut(s_rows))
                .enumerate()
                .for_each(work),
        }
    }
    sc.new_dry[0] = true;
    sc.new_dry[ny + 1] = true;
    std::mem::swap(&mut sc.new_dry, &mut dry);
    sc.row_dry = dry;
    let accs = sc.strips.iter().flat_map(|b| b.acc.iter().copied());
    finalize(accs, top, bot, &sc.sum_s_rows)
}
