//! Throughput bench: `cargo run --release --example gpubench`.
use itr_core::model::MonitorSet;
use itr_core::scenario::Hyetograph;
use itr_gpu::GpuSolver;
use itr_hydro::validation::{params, terrain};
use itr_hydro::{PreparedTerrain, Solver, Workspace};
use itr_core::model::NoopObserver;
use std::time::Instant;

fn main() {
    let mut g = match GpuSolver::new() {
        Ok(g) => g,
        Err(e) => return println!("no GPU: {e}"),
    };
    println!("adapter: {}", g.adapter_info());
    let mon = MonitorSet::default();
    for (label, n, cands, dur) in [("512^2 x1", 512usize, 1usize, 120.0), ("128^2 x16", 128, 16, 120.0), ("128^2 x64", 128, 64, 120.0), ("1024^2 x1", 1024, 1, 40.0)] {
        let base = terrain::<f32>(n, n, 2.0, |x, y| 5.0 - 0.01 * x + 0.3 * (y / 40.0).sin(), |_, _| false, 0.03, &[]);
        let mut p = params(dur, dur / 4.0);
        p.rain = Hyetograph::from_mm_h(&[[0.0, 100.0], [1e9, 100.0]]);
        p.initial_stage = Some(2.0);
        let ts: Vec<PreparedTerrain<f32>> = (0..cands).map(|_| PreparedTerrain::new(base.clone())).collect();
        let refs: Vec<&PreparedTerrain<f32>> = ts.iter().collect();
        g.run_batch(&p, &refs, &mon).unwrap(); // warm-up
        let t0 = Instant::now();
        let r = g.run_batch(&p, &refs, &mon).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let updates: u64 = r.iter().map(|x| x.steps * (n * n) as u64).sum();
        println!("{label}: {} steps, {:.3} s, {:.3} G cell-updates/s (useful), {:.1} us/step/batch", r[0].steps, dt, updates as f64 / dt / 1e9, dt / r[0].steps as f64 * 1e6);
        if n == 512 {
            let s = Solver::<f32>::new(p.clone());
            let mut ws = Workspace::<f32>::new(base.layout, &mon, 1, 32);
            let t0 = Instant::now();
            let cr = s.run_impl(&mut ws, &ts[0], None, &mut NoopObserver).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            println!("  CPU f32 single thread: {:.3} s, {:.3} G cell-updates/s ({} steps)", dt, cr.steps as f64 * (n * n) as f64 / dt / 1e9, cr.steps);
        }
    }
}
