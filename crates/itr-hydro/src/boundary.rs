//! Ghost-cell boundary conditions (spec §4.2). Filled once per step from the old state.
//!
//! Internal momentum convention: `qx` is +column (east), `qy` is +row (south).

use crate::aligned::Layout;
use crate::terrain::{Ghost, TerrainBase};
use itr_core::Real;

/// Fill ghost cells; returns per-padded-row "ghost wet" flags (west/east ghosts) and
/// whether the top/bottom ghost rows contain water.
#[allow(clippy::too_many_arguments)]
pub fn fill_ghosts<T: Real>(
    l: &Layout,
    terr: &TerrainBase<T>,
    z_level: &dyn Fn(usize, usize) -> f64,
    t: f64,
    inflow_scale: f64,
    h: &mut [T],
    qx: &mut [T],
    qy: &mut [T],
    ghost_wet: &mut [bool],
) {
    let (nx, ny) = (l.nx, l.ny);
    let g = terr.g;
    let b = &terr.boundaries;
    let inflow_q = |idx: usize, inv_len: f64| b.hydrographs[idx].rate(t) * inv_len * inflow_scale;
    // Critical depth (q²/g)^{1/3} via the shared multiplication-only inverse cube root.
    let crit = |q: f64| -> f64 {
        let x = q * q / g;
        if x > 0.0 { x * x.rcbrt() * x.rcbrt() } else { 0.0 }
    };
    for v in ghost_wet.iter_mut() {
        *v = false;
    }
    // West/east ghosts (x-normal).
    for j in 1..=ny {
        for (gi, ii, inward, spec) in [(0usize, 1usize, 1.0f64, &b.west[j - 1]), (nx + 1, nx, -1.0, &b.east[j - 1])] {
            let (gk, ik) = (l.at(gi, j), l.at(ii, j));
            let (hi, qxi, qyi) = (h[ik], qx[ik], qy[ik]);
            let (hg, qxg, qyg) = match spec {
                Ghost::Wall => (hi, -qxi, qyi),
                Ghost::Transmissive => (hi, qxi, qyi),
                Ghost::Stage(s) => (T::from_f64((s - z_level(ii - 1, j - 1)).max(0.0)), qxi, qyi),
                Ghost::Inflow(idx, inv_len) => {
                    let q = inflow_q(*idx, *inv_len);
                    let hg = hi.to_f64().max(crit(q));
                    (T::from_f64(hg), T::from_f64(inward * q), T::ZERO)
                }
            };
            h[gk] = hg;
            qx[gk] = qxg;
            qy[gk] = qyg;
            if hg > T::ZERO {
                ghost_wet[j] = true;
            }
        }
    }
    // North/south ghost rows (y-normal; +row is south).
    for i in 1..=nx {
        for (gj, ij, inward, spec) in [(0usize, 1usize, 1.0f64, &b.north[i - 1]), (ny + 1, ny, -1.0, &b.south[i - 1])] {
            let (gk, ik) = (l.at(i, gj), l.at(i, ij));
            let (hi, qxi, qyi) = (h[ik], qx[ik], qy[ik]);
            let (hg, qxg, qyg) = match spec {
                Ghost::Wall => (hi, qxi, -qyi),
                Ghost::Transmissive => (hi, qxi, qyi),
                Ghost::Stage(s) => (T::from_f64((s - z_level(i - 1, ij - 1)).max(0.0)), qxi, qyi),
                Ghost::Inflow(idx, inv_len) => {
                    let q = inflow_q(*idx, *inv_len);
                    let hg = hi.to_f64().max(crit(q));
                    (T::from_f64(hg), T::ZERO, T::from_f64(inward * q))
                }
            };
            h[gk] = hg;
            qx[gk] = qxg;
            qy[gk] = qyg;
            if hg > T::ZERO {
                ghost_wet[gj] = true;
            }
        }
    }
    for (i, j) in [(0, 0), (nx + 1, 0), (0, ny + 1), (nx + 1, ny + 1)] {
        let k = l.at(i, j);
        h[k] = T::ZERO;
        qx[k] = T::ZERO;
        qy[k] = T::ZERO;
    }
}
