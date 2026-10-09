//! Shared per-face and per-cell numerics (spec §5.2.1, §7.11.1).
//!
//! Both the scalar oracle and the fused kernel call exactly these functions, in the same
//! order, so their results can be compared bitwise. Divide/sqrt budget per cell-update:
//! velocity 1 div, `c` 1 sqrt, 2 faces × (1 sqrt + 1 div), friction `|q|` 1 sqrt +
//! 1 div, post-update CFL `c` 1 sqrt → 9 (8 for the scheme + 1 for the exact per-cell
//! CFL rate). The friction term uses a multiplication-only `h^{-1/3}`.

use itr_core::Real;

/// Per-step constants.
#[derive(Clone, Copy, Debug)]
pub struct StepConsts<T: Real> {
    pub g: T,
    pub g_half: T,
    pub dt: T,
    pub dtdx: T,
    pub dtdy: T,
    pub inv_dx: T,
    pub inv_dy: T,
    pub h_dry: T,
    pub heps2: T,
    pub q_min: T,
    pub rain: T,
    pub cap: T,
}

/// Velocities and celerity of one cell (old state).
#[derive(Clone, Copy, Debug, Default)]
pub struct Vel<T: Real> {
    pub u: T,
    pub v: T,
    pub c: T,
}

/// Desingularized velocity `u = 2h·q / (h² + max(h², h_ε²))` and `c = √(g h)`.
#[inline(always)]
pub fn cell_vel<T: Real>(h: T, qx: T, qy: T, k: &StepConsts<T>) -> Vel<T> {
    let h2 = h * h;
    let inv = T::ONE / (h2 + h2.max(k.heps2));
    let t = (h + h) * inv;
    Vel { u: qx * t, v: qy * t, c: (k.g * h).sqrt() }
}

/// Face flux tuple: mass, normal momentum seen by the left (low-index) cell, normal
/// momentum seen by the right cell (both with the hydrostatic-reconstruction pressure
/// correction folded in, `F_n − g/2·h*²`), and tangential momentum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Face<T: Real> {
    pub m: T,
    pub n_l: T,
    pub n_r: T,
    pub t: T,
}

/// HLL flux with two-rarefaction wave-speed estimates on reconstructed states.
/// `un`/`ut` are normal/tangential velocities.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hll<T: Real>(hl: T, ul: T, tl: T, cl: T, hr: T, ur: T, tr: T, cr: T, k: &StepConsts<T>) -> Face<T> {
    let z = T::ZERO;
    let half = T::from_f64(0.5);
    let quarter = T::from_f64(0.25);
    let two = T::from_f64(2.0);
    let dry_l = !(hl > z);
    let dry_r = !(hr > z);
    let ustar = half * (ul + ur) + cl - cr;
    let cstar = half * (cl + cr) + quarter * (ul - ur);
    let slw = (ul - cl).min(ustar - cstar);
    let srw = (ur + cr).max(ustar + cstar);
    let sl = if dry_l { ur - two * cr } else if dry_r { ul - cl } else { slw };
    let sr = if dry_l { ur + cr } else if dry_r { ul + two * cl } else { srw };
    let ml = hl * ul;
    let mr = hr * ur;
    let pl_ = k.g_half * hl * hl;
    let pr_ = k.g_half * hr * hr;
    let pl = ml * ul + pl_;
    let pr = mr * ur + pr_;
    // Branch-free selection (vectorizable): compute the star-region flux with a guarded
    // denominator, then select. Dry–dry faces return exactly zero.
    let both_dry = dry_l & dry_r;
    let use_l = sl >= z;
    let use_r = sr <= z;
    let den = if both_dry | use_l | use_r { T::ONE } else { sr - sl };
    let inv = T::ONE / den;
    let slsr = sl * sr;
    let fm_star = (sr * ml - sl * mr + slsr * (hr - hl)) * inv;
    let fn_star = (sr * pl - sl * pr + slsr * (mr - ml)) * inv;
    let fm = if both_dry { z } else if use_l { ml } else if use_r { mr } else { fm_star };
    let fn_ = if both_dry { z } else if use_l { pl } else if use_r { pr } else { fn_star };
    let ft = fm * if fm > z { tl } else { tr };
    Face { m: fm, n_l: fn_ - pl_, n_r: fn_ - pr_, t: ft }
}

/// Face in **signed-jump** mode: `dz = z_R − z_L`; exactly one side is reduced, so only
/// one square root is needed (the other side reuses the cell's `c`).
#[inline(always)]
pub fn face_signed<T: Real>(hl: T, vl: Vel<T>, hr: T, vr: Vel<T>, dz: T, normal_x: bool, k: &StepConsts<T>) -> Face<T> {
    let z = T::ZERO;
    let pos = dz > z;
    let hsl = (hl - dz.max(z)).max(z);
    let hsr = (hr - (-dz).max(z)).max(z);
    let hred = if pos { hsl } else { hsr };
    let cred = (k.g * hred).sqrt();
    let csl = if pos { cred } else { vl.c };
    let csr = if pos { vr.c } else { cred };
    if normal_x {
        hll(hsl, vl.u, vl.v, csl, hsr, vr.u, vr.v, csr, k)
    } else {
        hll(hsl, vl.v, vl.u, csl, hsr, vr.v, vr.u, csr, k)
    }
}

/// Face in **crest** mode: non-negative jumps `d_l = z_f − z_L`, `d_r = z_f − z_R`
/// (face-crest coarse levels, spec §19.1). Two square roots.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn face_crest<T: Real>(hl: T, vl: Vel<T>, hr: T, vr: Vel<T>, d_l: T, d_r: T, normal_x: bool, k: &StepConsts<T>) -> Face<T> {
    let z = T::ZERO;
    let hsl = (hl - d_l).max(z);
    let hsr = (hr - d_r).max(z);
    let csl = if d_l > z { (k.g * hsl).sqrt() } else { vl.c };
    let csr = if d_r > z { (k.g * hsr).sqrt() } else { vr.c };
    if normal_x {
        hll(hsl, vl.u, vl.v, csl, hsr, vr.u, vr.v, csr, k)
    } else {
        hll(hsl, vl.v, vl.u, csl, hsr, vr.v, vr.u, csr, k)
    }
}

/// Result of one cell update.
#[derive(Clone, Copy, Debug)]
pub struct CellOut<T: Real> {
    pub h: T,
    pub qx: T,
    pub qy: T,
    /// CFL rate (|u|+c)/dx + (|v|+c)/dy of the new state.
    pub s: T,
    pub infil: T,
    pub fix: T,
    pub bad: bool,
}

/// Finite-volume flux update of one cell. `w`/`e` are x-faces west/east of the cell,
/// `n`/`s` are y-faces north (row−1)/south.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn flux_update<T: Real>(h: T, qx: T, qy: T, w: &Face<T>, e: &Face<T>, n: &Face<T>, s: &Face<T>, k: &StepConsts<T>) -> (T, T, T) {
    let h1 = h - k.dtdx * (e.m - w.m) - k.dtdy * (s.m - n.m);
    let qx1 = qx - k.dtdx * (e.n_l - w.n_r) - k.dtdy * (s.t - n.t);
    let qy1 = qy - k.dtdx * (e.t - w.t) - k.dtdy * (s.n_l - n.n_r);
    (h1, qx1, qy1)
}

/// Sources after the flux update, in fixed order: positivity guard → rain →
/// infiltration → point-implicit Manning friction → drying → subnormal flush.
#[inline(always)]
pub fn sources<T: Real>(h1: T, qx1: T, qy1: T, wet: T, gn2: T, k: &StepConsts<T>) -> CellOut<T> {
    let z = T::ZERO;
    let mut h1 = h1;
    // Roundoff-negative depths are set to zero and the added volume is reported
    // explicitly in the ledger. Anything beyond roundoff is flagged unhealthy.
    let bad = !(h1 >= -k.h_dry);
    let fix = if h1 < z { -h1 } else { z };
    if h1 < z {
        h1 = z;
    }
    let h2 = h1 + k.rain * wet;
    let infil = (k.cap * wet).min(h2);
    let h3 = h2 - infil;
    let dry = h3 < k.h_dry;
    let r = (if dry { T::ONE } else { h3 }).rcbrt();
    let r3 = r * r * r;
    let r7 = r3 * r3 * r;
    let qm = (qx1 * qx1 + qy1 * qy1).sqrt();
    let inv = T::ONE / (T::ONE + k.dt * gn2 * qm * r7);
    // Mask by multiplication, not a select on the quotient: a select lets LLVM sink the
    // division (and the gn2 load) into a conditional block, which blocks vectorization
    // on targets without masked loads. `inv` is always finite (r = 1 when dry), so
    // ×1 is exact and ×0 gives ±0, canonicalized below.
    let keep = if dry { z } else { T::ONE };
    let mut qx2 = qx1 * inv * keep;
    let mut qy2 = qy1 * inv * keep;
    if qx2.abs() < k.q_min {
        qx2 = z;
    }
    if qy2.abs() < k.q_min {
        qy2 = z;
    }
    let c = (k.g * h3).sqrt();
    let s_rate = (qx2.abs() * r3 + c) * k.inv_dx + (qy2.abs() * r3 + c) * k.inv_dy;
    #[allow(clippy::eq_op)] // NaN test without a libm/float-classify call
    let bad = bad || !(qm == qm);
    CellOut { h: h3.canon_zero(), qx: qx2.canon_zero(), qy: qy2.canon_zero(), s: s_rate, infil, fix, bad }
}

/// Full cell update = `flux_update` then `sources` (reference composition).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn update_cell<T: Real>(
    h: T,
    qx: T,
    qy: T,
    w: &Face<T>,
    e: &Face<T>,
    n: &Face<T>,
    s: &Face<T>,
    wet: T,
    gn2: T,
    k: &StepConsts<T>,
) -> CellOut<T> {
    let (h1, qx1, qy1) = flux_update(h, qx, qy, w, e, n, s, k);
    sources(h1, qx1, qy1, wet, gn2, k)
}
