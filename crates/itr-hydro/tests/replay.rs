//! §9.1.1: baseline-locked Δt and subdomain replay.

use itr_core::design::TerrainEdit;
use itr_core::model::{AbortReason, MonitorSet, NoopObserver};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_core::Real;
use itr_hydro::replay::{sub_terrain, Roi, Tape};
use itr_hydro::validation::{params, terrain};
use itr_hydro::{BaselineRecord, PreparedTerrain, Solver, SolverParams, TerrainBase, Workspace};
use std::sync::Arc;

const NX: usize = 72;
const NY: usize = 56;

fn base<T: Real>() -> Arc<TerrainBase<T>> {
    let segs = [
        BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 0.0], [200.0, 2.0], [600.0, 0.3]], range_m: Some([40.0, 70.0]) },
        BoundarySegment::Transmissive { side: Side::East, range_m: None },
    ];
    terrain::<T>(NX, NY, 2.0, |x, y| 50.0 + 0.004 * (150.0 - x) + 0.3 * (x / 11.0).sin() * (y / 9.0).cos(), |_, _| false, 0.035, &segs)
}

fn p() -> SolverParams {
    let mut p = params(800.0, 40.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 0.0], [100.0, 0.0], [200.0, 60.0], [500.0, 0.0]]);
    p
}

const ROI: Roi = Roi { c0: 20, r0: 14, w: 30, h: 26 };

fn roi_monitors() -> MonitorSet {
    MonitorSet::new((0..(NX * NY) as u32).filter(|&c| ROI.contains_cell(c, NX) && c % 5 == 1).collect())
}

/// Baseline (adaptive) with Δt and tape recording.
fn baseline<T: Real>(b: &Arc<TerrainBase<T>>) -> (itr_core::model::RunSummary, Workspace<T>, BaselineRecord<T>) {
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<T>::new(b.layout, &roi_monitors(), 1, 32);
    let rec = BaselineRecord { tape: Some(Tape::new(ROI, &b.layout, 1 << 30)), ..Default::default() };
    ws.record = Some((0, rec));
    let s = Solver::<T>::new(p()).run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
    let rec = ws.take_record().unwrap();
    (s, ws, rec)
}

fn bits<T: Real>(v: &[T]) -> Vec<u64> {
    v.iter().map(|x| x.to_f64().to_bits()).collect()
}

fn suite<T: Real>() {
    let b = base::<T>();
    let (s0, ws0, rec) = baseline(&b);
    assert!(s0.aborted.is_none());
    let dts = Arc::new(rec.dts.clone());
    assert_eq!(dts.len() as u64, s0.steps);

    // Locked Δt, zero edit: bitwise identical to the baseline.
    let mut pl = p();
    pl.locked_dt = Some(dts.clone());
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<T>::new(b.layout, &roi_monitors(), 1, 32);
    let s1 = Solver::<T>::new(pl.clone()).run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
    assert_eq!(bits(&ws.h), bits(&ws0.h), "locked zero-edit h differs");
    assert_eq!(s1.steps, s0.steps);
    assert!(s1.dt_fallback_t.is_none());

    // Locked Δt with an edit that speeds flow up enough to break the hard CFL limit:
    // the run falls back to adaptive Δt from a sync snapshot and stays healthy.
    let mut pf = pl.clone();
    pf.cfl_hard = 0.36; // tight margin over the baseline's C = 0.35 so any speed-up trips it
    let w = 30;
    let e = TerrainEdit { c0: 0, r0: 0, w, h: NY, dz: (0..w * NY).map(|k| -0.6 * ((k % w) as f64 / w as f64)).collect(), ..Default::default() };
    let mut te = PreparedTerrain::new(b.clone());
    te.apply_edit(&e);
    let mut wsf = Workspace::<T>::new(b.layout, &roi_monitors(), 1, 32);
    let sf = Solver::<T>::new(pf).run_impl(&mut wsf, &te, None, &mut NoopObserver).unwrap();
    assert!(sf.aborted.is_none());
    assert!(sf.dt_fallback_t.is_some(), "expected a locked-Δt fallback");
    let tol = if T::NAME == "f64" { 1e-10 } else { 1e-4 };
    assert!(sf.ledger.rel_residual < tol, "fallback ledger residual {}", sf.ledger.rel_residual);

    // Subdomain replay, zero edit: bitwise identical to the baseline restricted to the ROI.
    let tape = Arc::new(rec.tape.unwrap());
    assert!(!tape.overflow);
    let sub_mon = MonitorSet { cells: roi_monitors().cells.iter().map(|&c| ROI.map_cell(c, NX)).collect() };
    let st = sub_terrain(&terr, ROI);
    let mut wr = Workspace::<T>::new(tape.sub_layout, &sub_mon, 1, 32);
    wr.replay = Some((tape.clone(), 2e-3, 1e-3));
    let sr = Solver::<T>::new(pl.clone()).run_impl(&mut wr, &st, None, &mut NoopObserver).unwrap();
    assert!(sr.aborted.is_none(), "{:?}", sr.aborted);
    let mb = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(mb(&sr.monitor_max), mb(&s0.monitor_max), "replay monitors differ");
    assert_eq!(mb(&sr.monitor_tmax), mb(&s0.monitor_tmax));
    let sl = tape.sub_layout;
    for j in 1..=ROI.h {
        for i in 1..=ROI.w {
            let (a, c) = (wr.h[sl.at(i, j)], ws0.h[b.layout.at(i + ROI.c0, j + ROI.r0)]);
            assert_eq!(a.to_f64().to_bits(), c.to_f64().to_bits(), "replay h differs at ({i},{j})");
        }
    }
    // Replay with an edit inside the ROI interior, far from the band: still valid.
    let small = TerrainEdit { c0: 34, r0: 26, w: 3, h: 3, dz: vec![0.02; 9], ..Default::default() };
    let mut ts = PreparedTerrain::new(b.clone());
    ts.apply_edit(&small);
    let mut wr2 = Workspace::<T>::new(tape.sub_layout, &sub_mon, 1, 32);
    wr2.replay = Some((tape.clone(), 2e-3, 1e-3));
    let s2 = Solver::<T>::new(pl.clone()).run_impl(&mut wr2, &sub_terrain(&ts, ROI), None, &mut NoopObserver).unwrap();
    assert!(s2.aborted.is_none(), "small interior edit flagged: {:?}", s2.aborted);

    // Validity detector: a tall berm across the ROI's downstream edge backs water up to
    // the band → flagged replay-invalid.
    let (bw, bh) = (2, ROI.h);
    let berm = TerrainEdit { c0: ROI.c0 + ROI.w - 3, r0: ROI.r0, w: bw, h: bh, dz: vec![1.5; bw * bh], ..Default::default() };
    let mut tb = PreparedTerrain::new(b.clone());
    tb.apply_edit(&berm);
    let mut wr3 = Workspace::<T>::new(tape.sub_layout, &sub_mon, 1, 32);
    wr3.replay = Some((tape.clone(), 2e-3, 1e-3));
    let s3 = Solver::<T>::new(pl).run_impl(&mut wr3, &sub_terrain(&tb, ROI), None, &mut NoopObserver).unwrap();
    assert!(matches!(s3.aborted, Some(AbortReason::ReplayInvalid { .. })), "backwater not detected: {:?}", s3.aborted);
}

#[test]
fn locked_dt_and_replay_f64() {
    suite::<f64>();
}

#[test]
fn locked_dt_and_replay_f32() {
    suite::<f32>();
}

/// §9.1.1 "Tail re-partitioning": a run moved from 1 → 4 → 2 threads mid-run (at sync
/// points) is bitwise identical to a single-threaded run.
fn repartition_suite<T: Real>() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let b = base::<T>();
    let terr = PreparedTerrain::new(b.clone());
    let mut w1 = Workspace::<T>::new(b.layout, &roi_monitors(), 1, 32);
    let s1 = Solver::<T>::new(p()).run_impl(&mut w1, &terr, None, &mut NoopObserver).unwrap();
    let mut w2 = Workspace::<T>::new(b.layout, &roi_monitors(), 1, 32);
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    w2.threads_hint = Some(Arc::new(move || match c.fetch_add(1, Ordering::Relaxed) {
        0..=2 => 1,
        3..=9 => 4,
        _ => 2,
    }));
    let s2 = Solver::<T>::new(p()).run_impl(&mut w2, &terr, None, &mut NoopObserver).unwrap();
    assert!(calls.load(Ordering::Relaxed) > 10);
    assert_eq!(bits(&w1.h), bits(&w2.h));
    assert_eq!(bits(&w1.qx), bits(&w2.qx));
    let mb = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(mb(&s1.monitor_max), mb(&s2.monitor_max));
    assert_eq!(mb(&s1.monitor_tmax), mb(&s2.monitor_tmax));
    assert_eq!(s1.steps, s2.steps);
}

#[test]
fn tail_repartition_is_bitwise() {
    repartition_suite::<f64>();
    repartition_suite::<f32>();
}
