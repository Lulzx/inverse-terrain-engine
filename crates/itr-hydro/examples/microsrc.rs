use itr_hydro::numerics::{sources, StepConsts};
use std::time::Instant;
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn run(h: &mut [f32], qx: &mut [f32], qy: &mut [f32], wet: &[f32], gn2: &[f32], a: &mut [f32], b: &mut [f32], c: &mut [f32], k: &StepConsts<f32>) -> bool {
    let m = h.len();
    let (qx, qy, wet, gn2, a, b, c) = (&mut qx[..m], &mut qy[..m], &wet[..m], &gn2[..m], &mut a[..m], &mut b[..m], &mut c[..m]);
    let mut bad = false;
    for i in 0..m {
        let o = sources(h[i], qx[i], qy[i], wet[i], gn2[i], k);
        h[i] = o.h; qx[i] = o.qx; qy[i] = o.qy; a[i] = o.infil; b[i] = o.fix; c[i] = o.s; bad |= o.bad;
    }
    bad
}
fn main() {
    let n = 512;
    let k = StepConsts::<f32> { g: 9.81, g_half: 4.905, dt: 0.05, dtdx: 0.025, dtdy: 0.025, inv_dx: 0.5, inv_dy: 0.5, h_dry: 1e-6, heps2: 1e-10, q_min: 1e-30, rain: 1e-6, cap: 0.0 };
    let mut h: Vec<f32> = (0..n).map(|i| 0.3 + 0.001 * i as f32).collect();
    let mut qx: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32).sin()).collect();
    let mut qy = qx.clone();
    let wet = vec![1.0f32; n]; let gn2 = vec![9.81 * 0.03 * 0.03; n];
    let (mut a, mut b, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let iters = 400_000;
    let t = Instant::now();
    let mut bad = false;
    let (h0, q0) = (h.clone(), qx.clone());
    for it in 0..iters { if it % 64 == 0 { h.copy_from_slice(&h0); qx.copy_from_slice(&q0); qy.copy_from_slice(&q0); } bad |= run(&mut h, &mut qx, &mut qy, &wet, &gn2, &mut a, &mut b, &mut c, &k); }
    let ns = t.elapsed().as_nanos() as f64 / (iters * n) as f64;
    let mut r = vec![0.0f32; n];
    let t2 = Instant::now();
    for _ in 0..iters { rc_pass(&h, &mut r); h[0] += 1e-9; }
    println!("rcbrt-only pass: {:.3} ns/cell", t2.elapsed().as_nanos() as f64 / (iters * n) as f64);
    println!("sources: {ns:.3} ns/cell ({:.1} cycles @4.5GHz) bad={bad} {}", ns * 4.5, h[7] + qx[3]);
}
// Split variant: pass A computes depths and r = h^{-1/3}; pass B the rest.
#[allow(dead_code)]
#[inline(never)]
pub fn rc_pass(h: &[f32], r: &mut [f32]) {
    use itr_core::Real;
    let m = r.len(); let h = &h[..m];
    for i in 0..m { r[i] = (if h[i] < 1e-6 { 1.0 } else { h[i] }).rcbrt(); }
}
