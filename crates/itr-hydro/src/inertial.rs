//! Local-inertial screening physics (Bates et al. 2010; spec §7.11.2). Used only as a
//! *fidelity level* for search screening, never for final ranking.
//!
//! Staggered storage in the same workspace arrays: `qx[at(i, j)]` is the discharge on the
//! **east face** of cell (i, j) (so `qx[at(0, j)]` is the west boundary face) and
//! `qy[at(i, j)]` the discharge on the **south face** (`qy[at(i, 0)]` is the north
//! boundary face); +qy points south (internal convention). Terrain enters only through
//! the face jumps, never `z + h`:
//!
//! h_f = max(h_L − d_L, h_R − d_R, 0),   ∂η/∂x ≈ ((h_R − d_R) − (h_L − d_L)) / Δx
//! q ← (q − g h_f Δt ∂η/∂x) / (1 + g n² Δt |q| h_f^{−7/3})
//!
//! where d_L, d_R are the rises from each cell's bed to the face bed (signed mode: from
//! the jump Δz; crest mode: the stored crest jumps). Δt = C_LI Δx / (√(g h) + |u|) with
//! C_LI = cfl/0.7 (0.5 at cfl = 0.35; the 2D staggered gravity-wave limit is 1/√2), i.e.
//! the CFL rate is `0.7 (√(g h) + |u|) / min(Δx, Δy)`.
//! Interior faces use de Almeida et al. (2012) θ-weighting of the old discharge (θ = 0.8).

use crate::aligned::Layout;
use crate::kernel::{Monitors, StepCtx, StepStats};
use crate::terrain::{FaceMode, Ghost};
use itr_core::Real;

#[inline]
fn rises<T: Real>(mode: FaceMode, f0: T, f1: T) -> (f64, f64) {
    match mode {
        FaceMode::Signed => {
            let dz = f0.to_f64();
            (dz.max(0.0), (-dz).max(0.0))
        }
        FaceMode::Crest => (f0.to_f64(), f1.to_f64()),
    }
}

/// Signed bed jump z_R − z_L of a face.
#[inline]
fn jump<T: Real>(mode: FaceMode, f0: T, f1: T) -> f64 {
    match mode {
        FaceMode::Signed => f0.to_f64(),
        FaceMode::Crest => f0.to_f64() - f1.to_f64(),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn face_q(q: f64, hl: f64, hr: f64, dl: f64, dr: f64, d: f64, gn2: f64, g: f64, dt: f64, h_dry: f64) -> f64 {
    let (el, er) = (hl - dl, hr - dr);
    let hf = el.max(er).max(0.0);
    if hf <= h_dry {
        return 0.0;
    }
    let slope = (er - el) / d;
    let r = hf.rcbrt();
    let r7 = r * r * r * r * r * r * r;
    (q - g * hf * dt * slope) / (1.0 + gn2 * dt * q.abs() * r7)
}

/// One local-inertial step on the workspace arrays (ghost depths already filled).
#[allow(clippy::too_many_arguments)]
pub fn step_local_inertial<T: Real>(
    cx: &StepCtx<T>,
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    mon: &mut Monitors,
    row_dry: &mut [bool],
    old: &mut Vec<f64>,
) -> StepStats {
    let l: &Layout = cx.l;
    let (nx, ny) = (l.nx, l.ny);
    let terr = cx.terr;
    let b = &terr.base;
    let (dx, dy, g, dt) = (b.dx, b.dy, b.g, cx.dt);
    let h_dry = cx.k.h_dry.to_f64();
    let mode = b.mode;
    let gn2 = |i: usize, j: usize| b.gn2.row(j)[i].to_f64();
    let hv = |h: &[T], i: usize, j: usize| h[l.at(i, j)].to_f64();
    let bd = &b.boundaries;
    // Old face discharges for de Almeida et al. (2012) θ-weighting (θ = 0.8), which
    // damps the checkerboard mode of the plain scheme on steep, thin-film terrain.
    let len = l.len();
    old.clear();
    old.extend(qx.iter().map(|v| v.to_f64()));
    old.extend(qy.iter().map(|v| v.to_f64()));
    let (ox, oy) = old.split_at(len);
    const TH: f64 = 0.8;
    let qtx = |i: usize, j: usize| {
        let c = ox[l.at(i, j)];
        if i >= 1 && i + 1 < nx { TH * c + 0.5 * (1.0 - TH) * (ox[l.at(i - 1, j)] + ox[l.at(i + 1, j)]) } else { c }
    };
    let qty = |i: usize, j: usize| {
        let c = oy[l.at(i, j)];
        if j >= 1 && j + 1 < ny { TH * c + 0.5 * (1.0 - TH) * (oy[l.at(i, j - 1)] + oy[l.at(i, j + 1)]) } else { c }
    };
    // x-faces: i = 0 (west boundary) ..= nx (east boundary).
    for j in 1..=ny {
        for i in 0..=nx {
            let k = l.at(i, j);
            let (f0, f1) = (terr.fx[0].row(j)[i], terr.fx[1].row(j)[i]);
            let q = if i == 0 || i == nx {
                let (spec, inward) = if i == 0 { (&bd.west[j - 1], 1.0) } else { (&bd.east[j - 1], -1.0) };
                match spec {
                    Ghost::Wall => 0.0,
                    // Zero depth gradient, bed slope extrapolated from the adjacent
                    // interior face (normal-depth outflow).
                    Ghost::Transmissive => {
                        let (ci, fi) = if i == 0 { (1, 1) } else { (nx, nx - 1) };
                        let dz = if nx > 1 { jump(mode, terr.fx[0].row(j)[fi], terr.fx[1].row(j)[fi]) } else { 0.0 };
                        // Outflow only: use the bed slope only where it falls outward,
                        // and never let the boundary face carry water inward.
                        let dz = if i == 0 { dz.max(0.0) } else { dz.min(0.0) };
                        let hc = hv(h, ci, j);
                        let q = face_q(qx[k].to_f64(), hc, hc, dz.max(0.0), (-dz).max(0.0), dx, gn2(ci, j), g, dt, h_dry);
                        if i == 0 { q.min(0.0) } else { q.max(0.0) }
                    }
                    // Ghost cell holds the inflow depth/discharge set by `fill_ghosts`.
                    Ghost::Inflow(..) => inward * qx[l.at(if i == 0 { 0 } else { nx + 1 }, j)].to_f64().abs(),
                    Ghost::Stage(_) => face_q(qx[k].to_f64(), hv(h, i, j), hv(h, i + 1, j), 0.0, 0.0, dx, gn2(i.max(1).min(nx), j), g, dt, h_dry),
                }
            } else {
                let (dl, dr) = rises(mode, f0, f1);
                let n2 = 0.5 * (gn2(i, j) + gn2(i + 1, j));
                face_q(qtx(i, j), hv(h, i, j), hv(h, i + 1, j), dl, dr, dx, n2, g, dt, h_dry)
            };
            qx[k] = T::from_f64(q).canon_zero();
        }
    }
    // y-faces: j = 0 (north boundary) ..= ny (south boundary); +q is southward.
    for j in 0..=ny {
        for i in 1..=nx {
            let k = l.at(i, j);
            let (f0, f1) = (terr.fy[0].row(j)[i], terr.fy[1].row(j)[i]);
            let q = if j == 0 || j == ny {
                let (spec, inward) = if j == 0 { (&bd.north[i - 1], 1.0) } else { (&bd.south[i - 1], -1.0) };
                match spec {
                    Ghost::Wall => 0.0,
                    Ghost::Transmissive => {
                        let (cj, fj) = if j == 0 { (1, 1) } else { (ny, ny - 1) };
                        let dz = if ny > 1 { jump(mode, terr.fy[0].row(fj)[i], terr.fy[1].row(fj)[i]) } else { 0.0 };
                        let dz = if j == 0 { dz.max(0.0) } else { dz.min(0.0) };
                        let hc = hv(h, i, cj);
                        let q = face_q(qy[k].to_f64(), hc, hc, dz.max(0.0), (-dz).max(0.0), dy, gn2(i, cj), g, dt, h_dry);
                        if j == 0 { q.min(0.0) } else { q.max(0.0) }
                    }
                    Ghost::Inflow(..) => inward * qy[l.at(i, if j == 0 { 0 } else { ny + 1 })].to_f64().abs(),
                    Ghost::Stage(_) => face_q(qy[k].to_f64(), hv(h, i, j), hv(h, i, j + 1), 0.0, 0.0, dy, gn2(i, j.max(1).min(ny)), g, dt, h_dry),
                }
            } else {
                let (dl, dr) = rises(mode, f0, f1);
                let n2 = 0.5 * (gn2(i, j) + gn2(i, j + 1));
                face_q(qty(i, j), hv(h, i, j), hv(h, i, j + 1), dl, dr, dy, n2, g, dt, h_dry)
            };
            qy[k] = T::from_f64(q).canon_zero();
        }
    }
    // Continuity + sources, ledger, CFL rate, monitors.
    let mut st = StepStats::default();
    let rain = cx.k.rain.to_f64();
    let cap = cx.k.cap.to_f64();
    let (rx, ry) = (dt / dx, dt / dy);
    for j in 1..=ny {
        let wet_row = b.wet.row(j);
        let mut any = false;
        for i in 1..=nx {
            let k = l.at(i, j);
            let wet = wet_row[i].to_f64();
            let (qw, qe) = (qx[l.at(i - 1, j)].to_f64(), qx[k].to_f64());
            let (qn, qs) = (qy[l.at(i, j - 1)].to_f64(), qy[k].to_f64());
            let mut hn = h[k].to_f64() + rx * (qw - qe) + ry * (qn - qs) + rain * wet;
            let infil = cap.min(hn.max(0.0)) * wet;
            hn -= infil;
            st.infil += infil;
            if hn < 0.0 {
                st.fix -= hn;
                hn = 0.0;
            }
            if !hn.is_finite() {
                st.bad = true;
            }
            h[k] = T::from_f64(hn * wet).canon_zero();
            any |= hn > 0.0;
            let s = if hn > h_dry {
                let u = 0.5 * (qw.abs() + qe.abs()) / hn;
                let v = 0.5 * (qn.abs() + qs.abs()) / hn;
                ((g * hn).sqrt() + u.max(v)) * 0.7 / dx.min(dy)
            } else {
                0.0
            };
            st.smax = st.smax.max(s);
            if cx.want_sum_s {
                st.sum_s += s;
            }
        }
        row_dry[j] = !any;
        // Boundary ledger (west/east faces of this row).
        let (w, e) = (qx[l.at(0, j)].to_f64() * dt * dy, qx[l.at(nx, j)].to_f64() * dt * dy);
        if w > 0.0 { st.bnd_in += w } else { st.bnd_out -= w }
        if e > 0.0 { st.bnd_out += e } else { st.bnd_in -= e }
        let (s0, s1) = (mon.row_start[j - 1], mon.row_start[j]);
        for m in s0..s1 {
            let col = mon.cells[m] as usize % nx;
            let v = h[l.at(col + 1, j)].to_f64();
            if v > mon.max[m] {
                mon.max[m] = v;
                mon.tmax[m] = cx.t_new;
            }
        }
    }
    for i in 1..=nx {
        let (n, s) = (qy[l.at(i, 0)].to_f64() * dt * dx, qy[l.at(i, ny)].to_f64() * dt * dx);
        if n > 0.0 { st.bnd_in += n } else { st.bnd_out -= n }
        if s > 0.0 { st.bnd_out += s } else { st.bnd_in -= s }
    }
    // Infiltration and fixes are depths; the caller multiplies by cell area.
    st
}

/// In staggered mode the west/north ghost slots hold boundary-face discharges, which
/// `fill_ghosts` would overwrite: save them before and restore them after (except at
/// inflow faces, whose ghost value is the prescribed discharge).
pub fn save_boundary_faces<T: Real>(l: &Layout, qx: &[T], qy: &[T], buf: &mut Vec<T>) {
    buf.clear();
    for j in 1..=l.ny {
        buf.push(qx[l.at(0, j)]);
    }
    for i in 1..=l.nx {
        buf.push(qy[l.at(i, 0)]);
    }
}

pub fn restore_boundary_faces<T: Real>(l: &Layout, b: &crate::terrain::Boundaries, qx: &mut [T], qy: &mut [T], buf: &[T]) {
    for j in 1..=l.ny {
        if !matches!(b.west[j - 1], Ghost::Inflow(..)) {
            qx[l.at(0, j)] = buf[j - 1];
        }
    }
    for i in 1..=l.nx {
        if !matches!(b.north[i - 1], Ghost::Inflow(..)) {
            qy[l.at(i, 0)] = buf[l.ny + i - 1];
        }
    }
}
