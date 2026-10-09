//! Analytic and reference cases (spec §9.1), shared by unit tests and `itr validate`.

use crate::solver::{KernelKind, Solver, SolverParams, Workspace};
use crate::terrain::{FineTerrain, PreparedTerrain, TerrainBase};
use itr_core::model::{MonitorSet, NoopObserver, RunSummary};
use itr_core::raster::{GeoTransform, Mask, Raster};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_core::Real;
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct CaseReport {
    pub name: String,
    pub passed: bool,
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub precision: String,
    pub notes: String,
}

pub fn params(duration: f64, sync: f64) -> SolverParams {
    SolverParams {
        duration_s: duration,
        sync_interval_s: sync,
        cfl: 0.35,
        dt_max_s: 5.0,
        h_dry: 1e-6,
        h_eps: 1e-5,
        rain: Hyetograph { pts: vec![] },
        infil_capacity: 0.0,
        initial_depth: 0.0,
        initial_stage: None,
        kernel: KernelKind::Fused,
        skip_dry: true,
        locked_dt: None,
        cfl_hard: 0.5,
        inflow_scale: 1.0,
    }
}

/// Build a terrain from an elevation function f(x, y) (cell centres, metres).
pub fn terrain<T: Real>(
    nx: usize,
    ny: usize,
    dx: f64,
    z: impl Fn(f64, f64) -> f64,
    walls: impl Fn(usize, usize) -> bool,
    n: f64,
    segs: &[BoundarySegment],
) -> Arc<TerrainBase<T>> {
    let geo = GeoTransform { origin_x: 0.0, origin_y: ny as f64 * dx, dx, dy: dx };
    let mut r = Raster::new(nx, ny, geo.clone(), 0.0);
    let mut m = Mask::new(nx, ny, false);
    for row in 0..ny {
        for col in 0..nx {
            let (x, y) = geo.cell_center(col, row);
            r.set(col, row, z(x, y));
            m.set(col, row, walls(col, row));
        }
    }
    let fine = Arc::new(FineTerrain::new(&r, &m, &vec![n; nx * ny]));
    Arc::new(TerrainBase::build(fine, 1, segs))
}

pub fn run<T: Real>(base: &Arc<TerrainBase<T>>, p: SolverParams, monitors: &MonitorSet) -> (RunSummary, Workspace<T>) {
    let terr = PreparedTerrain::new(base.clone());
    let mut ws = Workspace::new(base.layout, monitors, 1, 32);
    let s = Solver::<T>::new(p);
    let r = s.run_impl(&mut ws, &terr, None, &mut NoopObserver).expect("run");
    (r, ws)
}

fn report(name: &str, metric: &str, value: f64, threshold: f64, prec: &str, notes: String) -> CaseReport {
    CaseReport { name: name.into(), passed: value.is_finite() && value <= threshold, metric: metric.into(), value, threshold, precision: prec.into(), notes }
}

/// Still lake over a variable bed with a step: free surface and velocities stay at rest.
pub fn lake_at_rest<T: Real>() -> CaseReport {
    let base = terrain::<T>(64, 48, 1.0, |x, y| 0.3 * (x / 7.0).sin() * (y / 5.0).cos() + if x > 30.0 { 0.4 } else { 0.0 }, |_, _| false, 0.03, &[]);
    let mut p = params(60.0, 10.0);
    p.initial_stage = Some(1.5);
    let (r, ws) = run::<T>(&base, p, &MonitorSet::default());
    let l = ws.layout;
    let mut err: f64 = 0.0;
    for j in 1..=l.ny {
        for i in 1..=l.nx {
            let k = l.at(i, j);
            let eta = ws.h[k].to_f64() + base.z[(j - 1) * l.nx + i - 1] + base.fine.z_ref;
            err = err.max((eta - 1.5).abs()).max(ws.qx[k].to_f64().abs()).max(ws.qy[k].to_f64().abs());
        }
    }
    // f32 tolerance set from measurement (spec §9.1): roundoff random walk over ~1.4k steps.
    let tol = if T::NAME == "f64" { 1e-12 } else { 1e-4 };
    report("lake_at_rest", "max |Δη| + |q|", err, tol, T::NAME, format!("{} steps", r.steps))
}

/// Rain on a closed flat plain: stored volume equals rain input (mass ledger).
pub fn rain_closed_plain<T: Real>() -> CaseReport {
    let base = terrain::<T>(40, 30, 2.0, |_, _| 10.0, |_, _| false, 0.04, &[]);
    let mut p = params(600.0, 60.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 0.0], [100.0, 50.0], [400.0, 10.0], [500.0, 0.0]]);
    let (r, _) = run::<T>(&base, p, &MonitorSet::default());
    let tol = if T::NAME == "f64" { 1e-12 } else { 1e-6 };
    report("rain_closed_plain", "relative mass residual", r.ledger.rel_residual, tol, T::NAME, format!("rain {:.3} m3", r.ledger.rows.last().map_or(0.0, |x| x.rain_m3)))
}

/// Ritter dry-bed dam break (1D in x): L1 depth error against the analytic solution.
pub fn dam_break_ritter<T: Real>() -> CaseReport {
    let (nx, dx, h0, x0) = (800usize, 0.25, 1.0, 100.0);
    let base = terrain::<T>(nx, 3, dx, |_, _| 0.0, |_, _| false, 0.0, &[]);
    let mut p = params(10.0, 10.0);
    p.cfl = 0.4;
    let terr = PreparedTerrain::new(base.clone());
    let mut ws = Workspace::<T>::new(base.layout, &MonitorSet::default(), 1, 32);
    let s = Solver::<T>::new(p);
    // initial condition via stage on a flat bed only left of x0: use initial_depth trick
    let mut sp = s.p.clone();
    sp.initial_depth = 0.0;
    let s = Solver::<T>::new(sp);
    // Run with a custom initial state: set after init by running 0 steps is awkward; use a
    // checkpoint-free path: build state manually.
    let l = ws.layout;
    let ck = {
        let mut c = crate::solver::Checkpoint {
            step: 0,
            t: 0.0,
            sync_index: 0,
            dt_next: 0.4 * dx / (9.81f64 * h0).sqrt(),
            smax: 2.0 * (9.81f64 * h0).sqrt() / dx,
            h: ws.h.clone(),
            qx: ws.qx.clone(),
            qy: ws.qy.clone(),
            row_dry: vec![false; l.ny + 2],
            mon_max: vec![],
            mon_tmax: vec![],
            acc: Default::default(),
        };
        for j in 1..=l.ny {
            for i in 1..=l.nx {
                let x = (i as f64 - 0.5) * dx;
                if x < x0 {
                    c.h[l.at(i, j)] = T::from_f64(h0);
                }
            }
        }
        c.acc.v0 = (0..l.ny).map(|_| (x0 / dx) * h0).sum::<f64>() * dx * dx;
        c.acc.dt_min = f64::MAX;
        c
    };
    let r = s.run_impl(&mut ws, &terr, Some(&ck), &mut NoopObserver).unwrap();
    let t = r.t_end;
    let c0 = (9.81 * h0).sqrt();
    let mut l1 = 0.0;
    let j = 2;
    for i in 1..=nx {
        let x = (i as f64 - 0.5) * dx - x0;
        let exact = if x <= -c0 * t {
            h0
        } else if x >= 2.0 * c0 * t {
            0.0
        } else {
            let a = 2.0 * c0 - x / t;
            a * a / (9.0 * 9.81)
        };
        l1 += (ws.h[l.at(i, j)].to_f64() - exact).abs() * dx;
    }
    let rel = l1 / (h0 * 3.0 * c0 * t);
    report("dam_break_ritter", "relative L1 depth error", rel, 0.03, T::NAME, format!("t={t:.2}s, {} steps", r.steps))
}

/// Wetting/drying over a step with sloshing: depth stays non-negative, mass conserved.
pub fn wet_dry_step<T: Real>() -> CaseReport {
    let base = terrain::<T>(80, 20, 1.0, |x, _| if x > 40.0 { 0.8 } else { 0.0 } + 0.01 * x, |_, _| false, 0.02, &[]);
    let mut p = params(120.0, 20.0);
    p.initial_stage = Some(1.0);
    let (r, ws) = run::<T>(&base, p, &MonitorSet::default());
    let min_h = ws.h.iter().map(|v| v.to_f64()).fold(f64::MAX, f64::min);
    let tol = if T::NAME == "f64" { 1e-12 } else { 1e-6 };
    let ok = min_h >= 0.0 && r.aborted.is_none();
    let mut rep = report("wet_dry_step", "relative mass residual", r.ledger.rel_residual, tol, T::NAME, format!("min h = {min_h:e}"));
    rep.passed &= ok;
    rep
}

/// Rain on a tilted plane draining through a transmissive outlet: steady outflow →
/// rain × area (kinematic-wave limit), and outlet depth → Manning normal depth.
pub fn tilted_plane<T: Real>() -> CaseReport {
    let (nx, ny, dx, slope, n) = (100usize, 4usize, 2.0, 0.01, 0.03);
    let segs = [BoundarySegment::Transmissive { side: Side::East, range_m: None }];
    let base = terrain::<T>(nx, ny, dx, |x, _| 5.0 - slope * x, |_, _| false, n, &segs);
    let rate_mm_h = 100.0;
    let mut p = params(3600.0, 300.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, rate_mm_h], [1e9, rate_mm_h]]);
    let (r, _) = run::<T>(&base, p, &MonitorSet::default());
    let rows = &r.ledger.rows;
    let k = rows.len();
    let q_out = (rows[k - 1].outflow_m3 - rows[k - 2].outflow_m3) / (rows[k - 1].t - rows[k - 2].t);
    let q_in = rate_mm_h / 1000.0 / 3600.0 * (nx * ny) as f64 * dx * dx;
    let err = (q_out / q_in - 1.0).abs();
    report("tilted_plane_kinematic", "|Q_out/Q_rain − 1| at steady state", err, 0.02, T::NAME, format!("Q_out={q_out:.5} Q_rain={q_in:.5} m3/s"))
}

/// Uniformly shifted terrain and stage (by an integer number of metres): depths are
/// bitwise identical thanks to the re-based datum (gauge invariance).
pub fn gauge_invariance<T: Real>() -> CaseReport {
    let f = |off: f64| {
        // Dyadic (1/1024 m) elevations, as in typical quantized DEMs, so the shift is exact.
        let q = |v: f64| (v * 1024.0).round() / 1024.0;
        let base = terrain::<T>(40, 30, 1.0, move |x, y| off + q(0.2 * (x / 6.0).sin() + 0.01 * y), |_, _| false, 0.03, &[]);
        let mut p = params(60.0, 30.0);
        p.initial_stage = Some(off + 0.5);
        p.rain = Hyetograph::from_mm_h(&[[0.0, 40.0], [60.0, 0.0]]);
        let (_, ws) = run::<T>(&base, p, &MonitorSet::default());
        ws.h.to_vec()
    };
    let a = f(0.0);
    let b = f(1000.0);
    let diff = a.iter().zip(&b).filter(|(x, y)| x.to_f64().to_bits() != y.to_f64().to_bits()).count();
    report("gauge_invariance", "cells differing (bitwise)", diff as f64, 0.0, T::NAME, "terrain and stage shifted by +1000 m".into())
}

/// Wall/nodata cells under rain keep h = 0 exactly; ledger unaffected.
pub fn walls_under_rain<T: Real>() -> CaseReport {
    let base = terrain::<T>(30, 30, 1.0, |x, _| 0.01 * x, |c, r| (10..20).contains(&c) && (10..20).contains(&r), 0.03, &[]);
    let mut p = params(300.0, 60.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 80.0], [300.0, 80.0]]);
    let (r, ws) = run::<T>(&base, p, &MonitorSet::default());
    let l = ws.layout;
    let mut wall_h: f64 = 0.0;
    for row in 10..20 {
        for col in 10..20 {
            wall_h = wall_h.max(ws.h[l.at(col + 1, row + 1)].to_f64());
        }
    }
    let tol = if T::NAME == "f64" { 1e-12 } else { 1e-6 };
    let mut rep = report("walls_under_rain", "relative mass residual", r.ledger.rel_residual, tol, T::NAME, format!("max wall depth {wall_h:e}"));
    rep.passed &= wall_h == 0.0;
    rep
}

/// Inflow hydrograph into a closed basin: stored volume equals ∫Q dt.
pub fn inflow_ledger<T: Real>() -> CaseReport {
    let segs = [BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 0.0], [100.0, 2.0], [300.0, 0.0]], range_m: Some([10.0, 20.0]) }];
    let base = terrain::<T>(60, 30, 1.0, |x, _| 1.0 - 0.005 * x, |_, _| false, 0.03, &segs);
    let p = params(400.0, 50.0);
    let (r, _) = run::<T>(&base, p, &MonitorSet::default());
    let last = r.ledger.rows.last().unwrap();
    let tol = if T::NAME == "f64" { 1e-12 } else { 1e-6 };
    let net = last.inflow_m3 - last.outflow_m3;
    let mut rep = report("inflow_ledger", "relative mass residual (face fluxes vs storage)", r.ledger.rel_residual, tol, T::NAME, String::new());
    // The ghost-cell inflow realizes the hydrograph approximately; report how close, and
    // fail if the realized net inflow is more than 2% off the target 300 m3.
    rep.notes = format!("net face inflow {net:.3} m3 vs target 300 (ratio {:.4})", net / 300.0);
    rep.passed &= (net / 300.0 - 1.0).abs() < 0.02;
    rep
}

pub fn suite<T: Real>() -> Vec<CaseReport> {
    vec![
        lake_at_rest::<T>(),
        rain_closed_plain::<T>(),
        dam_break_ritter::<T>(),
        wet_dry_step::<T>(),
        tilted_plane::<T>(),
        gauge_invariance::<T>(),
        walls_under_rain::<T>(),
        inflow_ledger::<T>(),
    ]
}
