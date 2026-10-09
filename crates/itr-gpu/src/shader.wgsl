// Shallow-water update kernel (spec 7.8), f32. Formulas mirror itr-hydro/src/numerics.rs.
// Baseline variant: `update` (one dispatch per step) + `finalize` (1 workgroup/candidate).

struct Globals {
    nx: u32, ny: u32, ncand: u32, n_mon: u32,
    g: f32, h_dry: f32, heps2: f32, q_min: f32,
    inv_dx: f32, inv_dy: f32, dx: f32, dy: f32,
    cfl: f32, dt_max: f32, duration: f32, sync: f32,
    sync_tol: f32, infil_cap: f32, n_sync: u32, n_rain: u32,
    n_cells: u32, has_mon: u32, pad0: u32, pad1: u32,
}

struct Ctl {
    t_hi: f32, t_lo: f32, dt_cfl: f32, dt: f32,
    tn_hi: f32, tn_lo: f32, dtdx: f32, dtdy: f32,
    rain: f32, cap: f32, sync_idx: u32, steps: u32,
    parity: u32, act: u32, flags: u32, hits: u32,
}

const F_DONE: u32 = 1u;
const F_BAD: u32 = 2u;

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;   // [half][q][cand][N]
@group(0) @binding(2) var<storage, read> dzx: array<f32>;           // [cand][ny][nx+1]
@group(0) @binding(3) var<storage, read> dzy: array<f32>;           // [cand][ny+1][nx]
@group(0) @binding(4) var<storage, read> cellp: array<vec2<f32>>;   // (wet, g n^2) per cell
@group(0) @binding(5) var<storage, read> bc: array<u32>;            // N | S | W | E ghost codes
@group(0) @binding(6) var<storage, read_write> ctl: array<Ctl>;
@group(0) @binding(7) var<storage, read_write> smaxbad: array<atomic<u32>>; // [cand][2]
@group(0) @binding(8) var<storage, read_write> mon: array<atomic<u32>>;     // [cand][n_mon]
@group(0) @binding(9) var<storage, read> mon_map: array<u32>;       // cell -> monitor+1
@group(0) @binding(10) var<storage, read> active_list: array<u32>;
@group(0) @binding(11) var<storage, read> rain_tab: array<vec2<f32>>;

fn qoff(half: u32, q: u32, cand: u32) -> u32 {
    return ((half * 3u + q) * G.ncand + cand) * G.n_cells;
}

// ---- numerics.rs ports -------------------------------------------------------------

fn rcbrt(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    var r = bitcast<f32>(0x54A23280u - bits / 3u);
    let x3 = x * 0.3333333432674408;
    let c43 = 1.3333333730697632;
    for (var i = 0; i < 3; i++) {
        let r2 = r * r;
        r = r * c43 - x3 * (r2 * r2);
    }
    return r;
}

fn hll(hl: f32, ul: f32, tl: f32, cl: f32, hr: f32, ur: f32, tr: f32, cr: f32) -> vec4<f32> {
    let dry_l = !(hl > 0.0);
    let dry_r = !(hr > 0.0);
    let ustar = 0.5 * (ul + ur) + cl - cr;
    let cstar = 0.5 * (cl + cr) + 0.25 * (ul - ur);
    let slw = min(ul - cl, ustar - cstar);
    let srw = max(ur + cr, ustar + cstar);
    let sl = select(select(slw, ul - cl, dry_r), ur - 2.0 * cr, dry_l);
    let sr = select(select(srw, ul + 2.0 * cl, dry_r), ur + cr, dry_l);
    let ml = hl * ul;
    let mr = hr * ur;
    let pl_ = 0.5 * G.g * hl * hl;
    let pr_ = 0.5 * G.g * hr * hr;
    let pl = ml * ul + pl_;
    let pr = mr * ur + pr_;
    let both_dry = dry_l && dry_r;
    let use_l = sl >= 0.0;
    let use_r = sr <= 0.0;
    let den = select(sr - sl, 1.0, both_dry || use_l || use_r);
    let inv = 1.0 / den;
    let slsr = sl * sr;
    let fm_star = (sr * ml - sl * mr + slsr * (hr - hl)) * inv;
    let fn_star = (sr * pl - sl * pr + slsr * (mr - ml)) * inv;
    let fm = select(select(select(fm_star, mr, use_r), ml, use_l), 0.0, both_dry);
    let fn_ = select(select(select(fn_star, pr, use_r), pl, use_l), 0.0, both_dry);
    let ft = fm * select(tr, tl, fm > 0.0);
    return vec4<f32>(fm, fn_ - pl_, fn_ - pr_, ft);
}

// vl/vr = (u, v, c). Result = (m, n_l, n_r, t).
fn face_signed(hl: f32, vl: vec3<f32>, hr: f32, vr: vec3<f32>, dz: f32, normal_x: bool) -> vec4<f32> {
    let pos = dz > 0.0;
    let hsl = max(hl - max(dz, 0.0), 0.0);
    let hsr = max(hr - max(-dz, 0.0), 0.0);
    let hred = select(hsr, hsl, pos);
    let cred = sqrt(G.g * hred);
    let csl = select(vl.z, cred, pos);
    let csr = select(cred, vr.z, pos);
    if normal_x {
        return hll(hsl, vl.x, vl.y, csl, hsr, vr.x, vr.y, csr);
    }
    return hll(hsl, vl.y, vl.x, csl, hsr, vr.y, vr.x, csr);
}

// ---- tile kernel ------------------------------------------------------------------

var<workgroup> s_h: array<f32, 324>;
var<workgroup> s_qx: array<f32, 324>;
var<workgroup> s_qy: array<f32, 324>;
var<workgroup> s_u: array<f32, 324>;
var<workgroup> s_v: array<f32, 324>;
var<workgroup> s_c: array<f32, 324>;
var<workgroup> f_x: array<vec4<f32>, 272>;
var<workgroup> f_y: array<vec4<f32>, 272>;
var<workgroup> wg_smax: atomic<u32>;

// Old state (h, qx, qy) of cell (gx, gy), with ghost cells for out-of-range indices.
fn load_cell(cand: u32, rd: u32, gx: i32, gy: i32) -> vec3<f32> {
    let nx = i32(G.nx);
    let ny = i32(G.ny);
    let out_x = gx < 0 || gx >= nx;
    let out_y = gy < 0 || gy >= ny;
    if out_x && out_y {
        return vec3<f32>(0.0);
    }
    let sx = clamp(gx, 0, nx - 1);
    let sy = clamp(gy, 0, ny - 1);
    let idx = u32(sy * nx + sx);
    var v = vec3<f32>(state[qoff(rd, 0u, cand) + idx], state[qoff(rd, 1u, cand) + idx], state[qoff(rd, 2u, cand) + idx]);
    if out_x {
        var code: u32;
        if gx < 0 { code = bc[2u * G.nx + u32(sy)]; } else { code = bc[2u * G.nx + G.ny + u32(sy)]; }
        if code == 0u { v.y = -v.y; }
    } else if out_y {
        var code: u32;
        if gy < 0 { code = bc[u32(sx)]; } else { code = bc[G.nx + u32(sx)]; }
        if code == 0u { v.z = -v.z; }
    }
    return v;
}

@compute @workgroup_size(16, 16, 1)
fn update(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
          @builtin(local_invocation_index) li: u32) {
    let cand = active_list[wid.z];
    let c = ctl[cand];
    let on = c.act == 1u;
    let rd = c.parity;
    let wr = 1u - rd;
    let gx0 = i32(wid.x) * 16;
    let gy0 = i32(wid.y) * 16;
    let nx = i32(G.nx);
    let ny = i32(G.ny);
    if li == 0u { atomicStore(&wg_smax, 0u); }

    // Phase 1: halo load + cell velocities (desingularized).
    if on {
        for (var e = li; e < 324u; e += 256u) {
            let hx = i32(e % 18u);
            let hy = i32(e / 18u);
            let v = load_cell(cand, rd, gx0 - 1 + hx, gy0 - 1 + hy);
            let h = v.x;
            let h2 = h * h;
            let inv = 1.0 / (h2 + max(h2, G.heps2));
            let t = (h + h) * inv;
            s_h[e] = h;
            s_qx[e] = v.y;
            s_qy[e] = v.z;
            s_u[e] = v.y * t;
            s_v[e] = v.z * t;
            s_c[e] = sqrt(G.g * h);
        }
    }
    workgroupBarrier();

    // Phase 2: faces. x-faces: 16 rows x 17; y-faces: 17 rows x 16.
    if on {
        let dzx_base = cand * u32(ny * (nx + 1));
        let dzy_base = cand * u32((ny + 1) * nx);
        for (var e = li; e < 272u; e += 256u) {
            let fy = e / 17u;
            let fx = e % 17u;
            let row = gy0 + i32(fy);
            let f = gx0 + i32(fx);
            if row < ny && f <= nx {
                let a = (fy + 1u) * 18u + fx;
                let dz = dzx[dzx_base + u32(row * (nx + 1) + f)];
                f_x[e] = face_signed(s_h[a], vec3<f32>(s_u[a], s_v[a], s_c[a]), s_h[a + 1u],
                                     vec3<f32>(s_u[a + 1u], s_v[a + 1u], s_c[a + 1u]), dz, true);
            }
        }
        for (var e = li; e < 272u; e += 256u) {
            let fy = e / 16u;
            let fx = e % 16u;
            let col = gx0 + i32(fx);
            let f = gy0 + i32(fy);
            if col < nx && f <= ny {
                let a = fy * 18u + fx + 1u;
                let b = a + 18u;
                let dz = dzy[dzy_base + u32(f * nx + col)];
                f_y[e] = face_signed(s_h[a], vec3<f32>(s_u[a], s_v[a], s_c[a]), s_h[b],
                                     vec3<f32>(s_u[b], s_v[b], s_c[b]), dz, false);
            }
        }
    }
    workgroupBarrier();

    // Phase 3: flux update + sources + friction.
    if on {
        let gx = gx0 + i32(lid.x);
        let gy = gy0 + i32(lid.y);
        if gx < nx && gy < ny {
            let lx = lid.x;
            let ly = lid.y;
            let w = f_x[ly * 17u + lx];
            let e = f_x[ly * 17u + lx + 1u];
            let n = f_y[ly * 16u + lx];
            let s = f_y[(ly + 1u) * 16u + lx];
            let a = (ly + 1u) * 18u + lx + 1u;
            let h = s_h[a];
            let h1a = h - c.dtdx * (e.x - w.x) - c.dtdy * (s.x - n.x);
            let qx1 = s_qx[a] - c.dtdx * (e.y - w.z) - c.dtdy * (s.w - n.w);
            let qy1 = s_qy[a] - c.dtdx * (e.w - w.w) - c.dtdy * (s.y - n.z);
            let idx = u32(gy * nx + gx);
            let cp = cellp[idx];
            var h1 = h1a;
            var bad = !(h1 >= -G.h_dry);
            if h1 < 0.0 { h1 = 0.0; }
            let h2 = h1 + c.rain * cp.x;
            let infil = min(c.cap * cp.x, h2);
            let h3 = h2 - infil;
            let dry = h3 < G.h_dry;
            let r = rcbrt(select(h3, 1.0, dry));
            let r3 = r * r * r;
            let r7 = r3 * r3 * r;
            let qm = sqrt(qx1 * qx1 + qy1 * qy1);
            let inv = 1.0 / (1.0 + c.dt * cp.y * qm * r7);
            let keep = select(1.0, 0.0, dry);
            var qx2 = qx1 * inv * keep;
            var qy2 = qy1 * inv * keep;
            if abs(qx2) < G.q_min { qx2 = 0.0; }
            if abs(qy2) < G.q_min { qy2 = 0.0; }
            let cc = sqrt(G.g * h3);
            let srate = (abs(qx2) * r3 + cc) * G.inv_dx + (abs(qy2) * r3 + cc) * G.inv_dy;
            bad = bad || !(qm == qm) || !(srate <= 3.0e38);
            let ho = select(h3, 0.0, h3 == 0.0);
            state[qoff(wr, 0u, cand) + idx] = ho;
            state[qoff(wr, 1u, cand) + idx] = select(qx2, 0.0, qx2 == 0.0);
            state[qoff(wr, 2u, cand) + idx] = select(qy2, 0.0, qy2 == 0.0);
            if bad {
                atomicOr(&smaxbad[cand * 2u + 1u], 1u);
            } else {
                atomicMax(&wg_smax, bitcast<u32>(abs(srate)));
            }
            if G.has_mon != 0u {
                let mi = mon_map[idx];
                if mi != 0u {
                    atomicMax(&mon[cand * G.n_mon + mi - 1u], bitcast<u32>(abs(ho)));
                }
            }
        }
    }
    workgroupBarrier();
    if on && li == 0u {
        atomicMax(&smaxbad[cand * 2u], atomicLoad(&wg_smax));
    }
}

// ---- finalize: dt, time advance, rain integral -------------------------------------

fn rain_rate(xs: f32, n: u32, ts: f32, tlo: f32) -> f32 {
    // Rate at relative time xs (time = t + xs) with piecewise-linear table, constant extrapolation.
    let p0 = rain_tab[0];
    let x0 = (p0.x - ts) - tlo;
    if xs <= x0 { return p0.y; }
    for (var k = 1u; k < n; k++) {
        let a = rain_tab[k - 1u];
        let b = rain_tab[k];
        let xb = (b.x - ts) - tlo;
        if xs <= xb {
            let xa = (a.x - ts) - tlo;
            if xb == xa { return b.y; }
            return a.y + (b.y - a.y) * (xs - xa) / (xb - xa);
        }
    }
    return rain_tab[n - 1u].y;
}

// Exact integral of the piecewise-linear rate over [t, t+dt] (relative coordinates).
fn rain_integral(ts: f32, tlo: f32, dt: f32) -> f32 {
    let n = G.n_rain;
    if n == 0u || !(dt > 0.0) { return 0.0; }
    var total = 0.0;
    var x = 0.0;
    var r = rain_rate(0.0, n, ts, tlo);
    for (var k = 0u; k < n; k++) {
        let xk = (rain_tab[k].x - ts) - tlo;
        if xk > 0.0 && xk < dt {
            let rk = rain_rate(xk, n, ts, tlo);
            total += 0.5 * (r + rk) * (xk - x);
            x = xk;
            r = rk;
        }
    }
    let rb = rain_rate(dt, n, ts, tlo);
    total += 0.5 * (r + rb) * (dt - x);
    return total;
}

@compute @workgroup_size(1)
fn finalize(@builtin(workgroup_id) wid: vec3<u32>) {
    let cand = active_list[wid.x];
    var c = ctl[cand];
    if (c.flags & F_DONE) != 0u {
        c.act = 0u;
        ctl[cand] = c;
        return;
    }
    if c.act == 1u {
        // The previous update ran: read its CFL rate, reset the slot, advance time.
        let sm = bitcast<f32>(atomicLoad(&smaxbad[cand * 2u]));
        let bad = atomicLoad(&smaxbad[cand * 2u + 1u]);
        atomicStore(&smaxbad[cand * 2u], 0u);
        if bad != 0u {
            c.flags = c.flags | F_DONE | F_BAD;
            c.act = 0u;
            ctl[cand] = c;
            return;
        }
        c.t_hi = c.tn_hi;
        c.t_lo = c.tn_lo;
        c.parity = 1u - c.parity;
        c.steps += 1u;
        if c.hits != 0u { c.sync_idx += 1u; }
        c.dt_cfl = select(G.dt_max, G.cfl / sm, sm > 0.0);
        if c.sync_idx >= G.n_sync || (c.t_hi + c.t_lo) >= G.duration {
            c.flags = c.flags | F_DONE;
            c.act = 0u;
            ctl[cand] = c;
            return;
        }
    }
    var dt = select(G.dt_max, c.dt_cfl, c.dt_cfl < G.dt_max);
    let t_sync = min(f32(c.sync_idx + 1u) * G.sync, G.duration);
    let remaining = (t_sync - c.t_hi) - c.t_lo;
    let hits = dt >= remaining - G.sync_tol;
    var tn_hi: f32;
    var tn_lo: f32;
    if hits {
        dt = remaining;
        tn_hi = t_sync;
        tn_lo = 0.0;
    } else {
        // double-single add (two-sum)
        let s = c.t_hi + dt;
        let bb = s - c.t_hi;
        let err = (c.t_hi - (s - bb)) + (dt - bb);
        let lo = c.t_lo + err;
        tn_hi = s + lo;
        tn_lo = lo - (tn_hi - s);
    }
    c.hits = select(0u, 1u, hits);
    c.dt = dt;
    c.tn_hi = tn_hi;
    c.tn_lo = tn_lo;
    c.dtdx = dt / G.dx;
    c.dtdy = dt / G.dy;
    c.rain = rain_integral(c.t_hi, c.t_lo, dt);
    c.cap = G.infil_cap * dt;
    c.act = 1u;
    ctl[cand] = c;
}
