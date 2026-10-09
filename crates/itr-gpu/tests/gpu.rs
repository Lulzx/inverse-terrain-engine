use itr_core::design::TerrainEdit;
use itr_core::model::{MonitorSet, NoopObserver};
use itr_core::scenario::{BoundarySegment, Hyetograph, Side};
use itr_gpu::GpuSolver;
use itr_hydro::validation::{params, terrain};
use itr_hydro::{PreparedTerrain, Solver, Workspace};

fn gpu() -> Option<GpuSolver> {
    match GpuSolver::new() {
        Ok(g) => {
            println!("adapter: {}", g.adapter_info());
            Some(g)
        }
        Err(e) => {
            println!("SKIP: no GPU adapter ({e})");
            None
        }
    }
}

#[test]
fn lake_at_rest() {
    let Some(mut g) = gpu() else { return };
    let base = terrain::<f32>(64, 48, 1.0, |x, y| 0.3 * (x / 7.0).sin() * (y / 5.0).cos() + if x > 30.0 { 0.4 } else { 0.0 }, |_, _| false, 0.03, &[]);
    let mut p = params(60.0, 10.0);
    p.initial_stage = Some(1.5);
    let t = PreparedTerrain::new(base.clone());
    let r = g.run_batch(&p, &[&t], &MonitorSet::default()).unwrap();
    let r = &r[0];
    assert!(!r.bad);
    let mut err: f64 = 0.0;
    for k in 0..64 * 48 {
        let eta = r.depth[k] as f64 + base.z[k] + base.fine.z_ref;
        err = err.max((eta - 1.5).abs());
    }
    println!("lake_at_rest: max |d eta| = {err:e}, steps {}", r.steps);
    assert!(err < 1e-4);
}

#[test]
fn rain_closed_plain_volume() {
    let Some(mut g) = gpu() else { return };
    let base = terrain::<f32>(40, 30, 2.0, |_, _| 10.0, |_, _| false, 0.04, &[]);
    let mut p = params(600.0, 60.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 0.0], [100.0, 50.0], [400.0, 10.0], [500.0, 0.0]]);
    let t = PreparedTerrain::new(base.clone());
    let r = &g.run_batch(&p, &[&t], &MonitorSet::default()).unwrap()[0];
    let expect = p.rain.integral(0.0, 600.0) * 4.0 * (40 * 30) as f64;
    let rel = (r.volume_m3 / expect - 1.0).abs();
    println!("rain volume: gpu {:.6} expected {expect:.6} rel {rel:e}, t_end {}", r.volume_m3, r.t_end);
    assert!(rel < 1e-4);
    assert!((r.t_end - 600.0).abs() < 1e-3);
}

#[test]
fn cpu_comparison_tilted_rain_outflow() {
    let Some(mut g) = gpu() else { return };
    let segs = [BoundarySegment::Transmissive { side: Side::East, range_m: None }];
    let (nx, ny) = (100usize, 8usize);
    let base = terrain::<f32>(nx, ny, 2.0, |x, y| 5.0 - 0.01 * x + 0.002 * y, |_, _| false, 0.03, &segs);
    let mut p = params(900.0, 100.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 120.0], [600.0, 60.0], [800.0, 0.0]]);
    let cells: Vec<u32> = [(10, 2), (30, 4), (50, 1), (70, 6), (90, 3), (99, 5), (99, 0)].iter().map(|&(c, r)| (r * nx + c) as u32).collect();
    let mon = MonitorSet::new(cells);
    let t = PreparedTerrain::new(base.clone());
    let gr = &g.run_batch(&p, &[&t], &mon).unwrap()[0];
    let s = Solver::<f32>::new(p.clone());
    let mut ws = Workspace::<f32>::new(base.layout, &mon, 1, 32);
    let cr = s.run_impl(&mut ws, &t, None, &mut NoopObserver).unwrap();
    let mut worst: f64 = 0.0;
    for (k, (a, b)) in gr.monitor_max.iter().zip(&cr.monitor_max).enumerate() {
        let rel = (a - b).abs() / b.abs().max(1e-6);
        println!("monitor {k}: gpu {a:.6e} cpu {b:.6e} rel {rel:.3e}");
        worst = worst.max(rel);
    }
    println!("D3 observed: max monitor rel diff {worst:.3e}; steps gpu {} cpu {}; t_end gpu {} cpu {}", gr.steps, cr.steps, gr.t_end, cr.t_end);
    assert!(worst < 0.02);
    assert!(!gr.bad);
}

#[test]
fn batch_matches_single_bitwise() {
    let Some(mut g) = gpu() else { return };
    let segs = [BoundarySegment::Transmissive { side: Side::South, range_m: None }];
    let (nx, ny) = (48usize, 40usize);
    let base = terrain::<f32>(nx, ny, 2.0, |x, y| 3.0 - 0.01 * y + 0.2 * (x / 9.0).sin(), |_, _| false, 0.035, &segs);
    let mut p = params(300.0, 50.0);
    p.rain = Hyetograph::from_mm_h(&[[0.0, 90.0], [200.0, 90.0], [250.0, 0.0]]);
    let mk = |c0: usize, r0: usize, w: usize, h: usize, amp: f64| {
        let mut t = PreparedTerrain::new(base.clone());
        let e = TerrainEdit { c0, r0, w, h, dz: (0..w * h).map(|i| amp * (1.0 + (i % 5) as f64 * 0.1)).collect(), ..Default::default() };
        t.apply_edit(&e);
        t
    };
    let t0 = PreparedTerrain::new(base.clone());
    let t1 = mk(10, 10, 8, 6, 0.3);
    let t2 = mk(25, 18, 10, 4, -0.15);
    let mon = MonitorSet::new(vec![(35 * nx + 5) as u32, (20 * nx + 20) as u32, (39 * nx + 30) as u32]);
    let batch = g.run_batch(&p, &[&t0, &t1, &t2], &mon).unwrap();
    for (i, t) in [&t0, &t1, &t2].iter().enumerate() {
        let single = &g.run_batch(&p, &[t], &mon).unwrap()[0];
        let b = &batch[i];
        assert_eq!(b.steps, single.steps, "cand {i} steps");
        assert_eq!(b.t_end.to_bits(), single.t_end.to_bits());
        assert!(b.depth.iter().zip(&single.depth).all(|(a, c)| a.to_bits() == c.to_bits()), "cand {i} depth differs");
        assert!(b.monitor_max.iter().zip(&single.monitor_max).all(|(a, c)| a.to_bits() == c.to_bits()), "cand {i} monitors differ");
        println!("cand {i}: steps {} bitwise identical to single run", b.steps);
    }
    assert!(batch[0].depth != batch[1].depth, "edits should change results");
}

#[test]
fn unsupported_modes_rejected() {
    let Some(mut g) = gpu() else { return };
    // Crest mode (coarse level) is rejected.
    let fine = terrain::<f32>(32, 32, 1.0, |_, _| 0.0, |_, _| false, 0.03, &[]).fine.clone();
    let coarse = std::sync::Arc::new(itr_hydro::TerrainBase::<f32>::build(fine, 2, &[]));
    let t = PreparedTerrain::new(coarse);
    assert!(g.run_batch(&params(10.0, 5.0), &[&t], &MonitorSet::default()).is_err());
    // Stage/Inflow boundaries are rejected.
    let segs = [BoundarySegment::Inflow { side: Side::West, hydrograph: vec![[0.0, 1.0]], range_m: None }];
    let base = terrain::<f32>(16, 16, 1.0, |_, _| 0.0, |_, _| false, 0.03, &segs);
    let t = PreparedTerrain::new(base);
    assert!(g.run_batch(&params(10.0, 5.0), &[&t], &MonitorSet::default()).is_err());
}
