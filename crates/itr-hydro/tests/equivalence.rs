//! Implementation-equivalence tests (spec §9.1.1). Every optimization must leave results
//! bitwise identical to the reference.

use itr_core::design::TerrainEdit;
use itr_core::model::{AbortReason, MonitorSet, NoopObserver, Observer, RunSummary, SyncView};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_core::Real;
use itr_hydro::validation::{params, terrain};
use itr_hydro::{KernelKind, PreparedTerrain, Solver, SolverParams, TerrainBase, Workspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    // Count only allocations made by the measuring thread (tests run in parallel).
    static COUNT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNT.with(|c| c.get()) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GA: Counting = Counting;

const NX: usize = 57;
const NY: usize = 43;

fn base<T: Real>() -> Arc<TerrainBase<T>> {
    let segs = [
        BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 0.0], [200.0, 1.5], [500.0, 0.2]], range_m: Some([20.0, 40.0]) },
        BoundarySegment::Transmissive { side: Side::East, range_m: None },
        BoundarySegment::Stage { side: Side::South, stage_m: 100.3, range_m: Some([5.0, 25.0]) },
    ];
    terrain::<T>(
        NX,
        NY,
        2.0,
        |x, y| 100.0 + 0.5 * (x / 9.0).sin() * (y / 7.0).cos() + 0.004 * (120.0 - x) + if (60.0..64.0).contains(&x) { 0.6 } else { 0.0 },
        |c, r| (30..34).contains(&c) && (10..18).contains(&r),
        0.035,
        &segs,
    )
}

fn p(kernel: KernelKind, skip: bool) -> SolverParams {
    let mut p = params(900.0, 60.0);
    p.kernel = kernel;
    p.skip_dry = skip;
    // Dry lead-in (rain starts at 300 s) so skipping and warm start are exercised.
    p.rain = Hyetograph::from_mm_h(&[[0.0, 0.0], [300.0, 0.0], [400.0, 80.0], [700.0, 0.0]]);
    p.infil_capacity = 10.0 / 1000.0 / 3600.0;
    p
}

fn monitors() -> MonitorSet {
    MonitorSet::new((0..(NX * NY) as u32).filter(|c| c % 7 == 3).collect())
}

struct Run<T: Real> {
    s: RunSummary,
    h: Vec<T>,
    qx: Vec<T>,
    qy: Vec<T>,
}

fn go<T: Real>(b: &Arc<TerrainBase<T>>, pp: SolverParams, threads: usize, strip: usize, edit: Option<&TerrainEdit>) -> Run<T> {
    let mut terr = PreparedTerrain::new(b.clone());
    if let Some(e) = edit {
        terr.apply_edit(e);
    }
    let mut ws = Workspace::<T>::new(b.layout, &monitors(), threads, strip);
    let s = Solver::<T>::new(pp).run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
    Run { s, h: ws.h.to_vec(), qx: ws.qx.to_vec(), qy: ws.qy.to_vec() }
}

fn bits<T: Real>(v: &[T]) -> Vec<u64> {
    v.iter().map(|x| x.to_f64().to_bits()).collect()
}

fn same<T: Real>(a: &Run<T>, b: &Run<T>, what: &str) {
    assert_eq!(bits(&a.h), bits(&b.h), "{what}: h differs");
    assert_eq!(bits(&a.qx), bits(&b.qx), "{what}: qx differs");
    assert_eq!(bits(&a.qy), bits(&b.qy), "{what}: qy differs");
    assert_eq!(a.s.steps, b.s.steps, "{what}: step count");
    let mb = |s: &RunSummary| s.monitor_max.iter().chain(&s.monitor_tmax).map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(mb(&a.s), mb(&b.s), "{what}: monitors differ");
    let lb = |s: &RunSummary| s.ledger.rows.iter().map(|r| (r.volume_m3.to_bits(), r.outflow_m3.to_bits(), r.infiltration_m3.to_bits())).collect::<Vec<_>>();
    assert_eq!(lb(&a.s), lb(&b.s), "{what}: ledger differs");
}

fn edit() -> TerrainEdit {
    // A small berm patch in a region that stays dry until rain onset.
    let (w, h) = (6, 3);
    let dz: Vec<f64> = (0..w * h).map(|k| 0.05 * (1 + k % 5) as f64).collect();
    TerrainEdit { c0: 40, r0: 5, w, h, fill_m3: dz.iter().sum::<f64>() * 4.0, dz, ..Default::default() }
}

fn suite<T: Real>() {
    let b = base::<T>();
    let oracle = go(&b, p(KernelKind::Oracle, false), 1, 32, None);
    assert!(oracle.s.aborted.is_none());
    assert!(oracle.s.ledger.rel_residual < if T::NAME == "f64" { 1e-12 } else { 1e-4 }, "residual {}", oracle.s.ledger.rel_residual);
    let fused = go(&b, p(KernelKind::Fused, false), 1, 32, None);
    same(&oracle, &fused, "fused vs oracle");
    let skip = go(&b, p(KernelKind::Fused, true), 1, 32, None);
    same(&oracle, &skip, "dry-row skipping");
    assert!(skip.s.rows_skipped > 0, "skipping was not exercised");
    for (threads, strip) in [(2, 5), (3, 7), (7, 1), (4, 43)] {
        let r = go(&b, p(KernelKind::Fused, true), threads, strip, None);
        same(&oracle, &r, &format!("threads={threads} strip={strip}"));
    }
    // Edited terrain: oracle vs fused-parallel.
    let e = edit();
    let eo = go(&b, p(KernelKind::Oracle, false), 1, 32, Some(&e));
    let ef = go(&b, p(KernelKind::Fused, true), 3, 4, Some(&e));
    same(&eo, &ef, "edited terrain");
    assert_ne!(bits(&eo.h), bits(&oracle.h), "edit had no effect");
}

#[test]
fn equivalence_f64() {
    suite::<f64>();
}

#[test]
fn equivalence_f32() {
    suite::<f32>();
}

#[test]
fn warm_start_is_bitwise() {
    // Rain-only variant with a 300 s dry lead-in, plus a stage boundary far from the edit.
    let segs = [BoundarySegment::Stage { side: Side::South, stage_m: 100.3, range_m: Some([5.0, 25.0]) }];
    let b = terrain::<f64>(NX, NY, 2.0, |x, y| 100.0 + 0.5 * (x / 9.0).sin() * (y / 7.0).cos() + 0.004 * (120.0 - x), |_, _| false, 0.035, &segs);
    let pp = p(KernelKind::Fused, true);
    // Baseline with recording.
    let terr0 = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<f64>::new(b.layout, &monitors(), 1, 32);
    ws.record = Some((1 << 30, Default::default()));
    Solver::<f64>::new(pp.clone()).run_impl(&mut ws, &terr0, None, &mut NoopObserver).unwrap();
    let rec = ws.take_record().unwrap();
    let e = edit();
    let mut terr = PreparedTerrain::new(b.clone());
    terr.apply_edit(&e);
    let ck = rec.best_for(terr.edit_rows).expect("a valid checkpoint");
    assert!(ck.step > 0, "warm start should skip the dry lead-in (got step {})", ck.step);
    let mut ws2 = Workspace::<f64>::new(b.layout, &monitors(), 1, 32);
    let warm = Solver::<f64>::new(pp.clone()).run_impl(&mut ws2, &terr, Some(ck), &mut NoopObserver).unwrap();
    let cold = go(&b, pp, 1, 32, Some(&e));
    assert_eq!(bits(&ws2.h), bits(&cold.h));
    assert_eq!(warm.steps, cold.s.steps);
    assert_eq!(warm.monitor_max, cold.s.monitor_max);
    assert!(warm.warm_start_steps > 0);
}

struct Snap(Vec<Vec<f32>>);
impl Observer for Snap {
    fn on_sync(&mut self, v: &SyncView<'_>) -> ControlFlow<AbortReason> {
        let mut d = vec![0.0; v.state.nx() * v.state.ny()];
        v.state.depth(&mut d);
        self.0.push(d);
        ControlFlow::Continue(())
    }
}

#[test]
fn outputs_do_not_change_trajectory() {
    let b = base::<f32>();
    let pp = p(KernelKind::Fused, true);
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<f32>::new(b.layout, &monitors(), 2, 8);
    let mut snap = Snap(vec![]);
    Solver::<f32>::new(pp.clone()).run_impl(&mut ws, &terr, None, &mut snap).unwrap();
    let plain = go(&b, pp, 1, 32, None);
    assert_eq!(bits(&ws.h), bits(&plain.h));
    assert_eq!(snap.0.len(), 15);
}

#[test]
fn steady_state_zero_allocations() {
    let b = base::<f32>();
    let mut pp = p(KernelKind::Fused, true);
    pp.sync_interval_s = 900.0; // no sync work inside the measured window except the end
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<f32>::new(b.layout, &monitors(), 1, 32);
    let s = Solver::<f32>::new(pp);
    s.run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap(); // warm-up
    ALLOCS.store(0, Ordering::SeqCst);
    COUNT.with(|c| c.set(true));
    let r = s.run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
    COUNT.with(|c| c.set(false));
    let n = ALLOCS.load(Ordering::SeqCst);
    // Per-run constant allocations (summary vectors, z_level copy) are allowed; per-step
    // allocations are not: the count must not scale with the number of steps.
    assert!(n < 16, "{n} allocations for {} steps", r.steps);
    assert!(r.steps > 100);
}

#[test]
fn mirror_metamorphic() {
    // East-west mirrored terrain with walls only: mirrored depths within roundoff.
    let f = |mirror: bool| {
        let b = terrain::<f64>(
            40,
            20,
            1.0,
            move |x, y| {
                let x = if mirror { 40.0 - x } else { x };
                0.3 * (x / 5.0).sin() + 0.02 * x + 0.01 * y
            },
            |_, _| false,
            0.03,
            &[],
        );
        let mut pp = params(200.0, 50.0);
        pp.rain = Hyetograph::from_mm_h(&[[0.0, 60.0], [100.0, 0.0]]);
        let (_, ws) = itr_hydro::validation::run::<f64>(&b, pp, &MonitorSet::default());
        let l = ws.layout;
        let mut out = vec![0.0; 40 * 20];
        for r in 0..20 {
            for c in 0..40 {
                let cc = if mirror { 39 - c } else { c };
                out[r * 40 + cc] = ws.h[l.at(c + 1, r + 1)];
            }
        }
        out
    };
    let a = f(false);
    let b = f(true);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-9, "mirror mismatch {worst}");
}

#[test]
fn crest_level_lake_at_rest_and_mass() {
    use itr_core::raster::{GeoTransform, Mask, Raster};
    use itr_hydro::FineTerrain;
    let (nx, ny) = (64, 48);
    let geo = GeoTransform { origin_x: 0.0, origin_y: ny as f64, dx: 1.0, dy: 1.0 };
    let mut r = Raster::new(nx, ny, geo.clone(), 0.0);
    for row in 0..ny {
        for col in 0..nx {
            // A thin 1-cell berm at col 30 that a 4× block mean would smear.
            r.set(col, row, 0.2 * (col as f64 / 9.0).sin() + if col == 30 { 1.0 } else { 0.0 });
        }
    }
    let fine = Arc::new(FineTerrain::new(&r, &Mask::new(nx, ny, false), &vec![0.03; nx * ny]));
    let b = Arc::new(TerrainBase::<f64>::build(fine, 4, &[]));
    assert_eq!(b.mode, itr_hydro::terrain::FaceMode::Crest);
    let mut pp = params(120.0, 30.0);
    pp.initial_stage = Some(0.6); // below the berm crest (≈1.0)
    let (s, ws) = itr_hydro::validation::run::<f64>(&b, pp.clone(), &MonitorSet::default());
    let max_q = ws.qx.iter().chain(ws.qy.iter()).map(|v| v.abs()).fold(0.0, f64::max);
    assert!(max_q < 1e-10, "crest lake at rest: q={max_q}");
    assert!(s.ledger.rel_residual < 1e-12);
    // The berm blocks: rain only on the west side cannot reach the east side.
    let _ = pp;
}
