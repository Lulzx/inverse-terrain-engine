//! Local-inertial screening physics (§7.11.2): well-balancing, mass conservation, the
//! kinematic tilted-plane limit, and agreement with HLL on peak depths.

use itr_core::model::MonitorSet;
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_hydro::validation::{params, run, terrain};
use itr_hydro::KernelKind;

#[test]
fn lake_at_rest() {
    let b = terrain::<f64>(48, 36, 1.0, |x, y| 0.3 * (x / 7.0).sin() * (y / 5.0).cos() + if x > 24.0 { 0.4 } else { 0.0 }, |_, _| false, 0.03, &[]);
    let mut p = params(120.0, 10.0);
    p.kernel = KernelKind::LocalInertial;
    p.initial_stage = Some(1.5);
    let (s, ws) = run(&b, p, &MonitorSet::default());
    let l = ws.layout;
    let mut err: f64 = 0.0;
    for j in 1..=l.ny {
        for i in 1..=l.nx {
            err = err.max(ws.qx[l.at(i, j)].abs()).max(ws.qy[l.at(i, j)].abs());
        }
    }
    assert!(s.aborted.is_none());
    assert!(err < 1e-12, "lake at rest discharge {err}");
}

#[test]
fn rain_mass_and_tilted_plane() {
    // Closed plain: mass.
    let b = terrain::<f64>(30, 20, 2.0, |x, _| 0.001 * x, |_, _| false, 0.03, &[]);
    let mut p = params(600.0, 60.0);
    p.kernel = KernelKind::LocalInertial;
    p.rain = Hyetograph::from_mm_h(&[[0.0, 60.0], [300.0, 60.0], [301.0, 0.0]]);
    let (s, _) = run(&b, p, &MonitorSet::default());
    assert!(s.ledger.rel_residual < 1e-12, "LI mass residual {}", s.ledger.rel_residual);

    // Tilted plane draining east: steady outflow = rain × area.
    let segs = [BoundarySegment::Transmissive { side: Side::East, range_m: None }];
    let (nx, ny, dx) = (60, 4, 2.0);
    let b = terrain::<f64>(nx, ny, dx, |x, _| 0.01 * (nx as f64 * dx - x), |_, _| false, 0.03, &segs);
    let mut p = params(3600.0, 60.0);
    p.kernel = KernelKind::LocalInertial;
    let rate = 50.0 / 1000.0 / 3600.0;
    p.rain = Hyetograph::from_mm_h(&[[0.0, 50.0]]);
    let (s, _) = run(&b, p, &MonitorSet::default());
    let r = &s.ledger.rows;
    let (a, z) = (&r[r.len() - 6], &r[r.len() - 1]);
    let q_out = (z.outflow_m3 - a.outflow_m3) / (z.t - a.t);
    let q_rain = rate * (nx * ny) as f64 * dx * dx;
    assert!((q_out / q_rain - 1.0).abs() < 0.02, "LI steady outflow {q_out} vs {q_rain}");
}

#[test]
fn peak_depths_track_hll() {
    // Rain + inflow over an undulating floodplain: LI peak depths at monitored cells
    // should track HLL closely (screening fidelity, RQ8).
    let segs = [
        BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 0.0], [300.0, 3.0], [900.0, 0.5]], range_m: Some([30.0, 60.0]) },
        BoundarySegment::Transmissive { side: Side::East, range_m: None },
    ];
    let b = terrain::<f64>(64, 48, 2.0, |x, y| 20.0 + 0.003 * (128.0 - x) + 0.2 * (x / 13.0).sin() * (y / 9.0).cos(), |_, _| false, 0.04, &segs);
    let mon = MonitorSet::new((0..64 * 48).filter(|c| c % 17 == 4).collect());
    let mut p = params(1200.0, 60.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 30.0], [600.0, 30.0], [601.0, 0.0]]);
    let (hll, _) = run(&b, p.clone(), &mon);
    p.kernel = KernelKind::LocalInertial;
    let (li, _) = run(&b, p, &mon);
    assert!(li.ledger.rel_residual < 1e-10);
    let (mut num, mut den) = (0.0, 0.0);
    for (a, c) in hll.monitor_max.iter().zip(&li.monitor_max) {
        num += (a - c).abs();
        den += a.abs();
    }
    let rel = num / den;
    println!("LI vs HLL relative L1 peak-depth difference {rel:.4}; steps HLL {} LI {}", hll.steps, li.steps);
    assert!(rel < 0.15, "LI peak depths deviate {rel}");
    assert!(li.steps < hll.steps, "LI should take larger steps");
}
