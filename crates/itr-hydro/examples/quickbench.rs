use itr_core::model::MonitorSet;
use itr_core::scenario::Hyetograph;
use itr_core::Real;
use itr_hydro::validation::{params, terrain};
use itr_hydro::{PreparedTerrain, Solver, Workspace};

fn bench<T: Real>(n: usize, threads: usize) {
    let b = terrain::<T>(n, n, 2.0, |x, y| 0.5 * (x / 30.0).sin() * (y / 20.0).cos() + 0.002 * (x + y), |_, _| false, 0.03, &[]);
    let mut p = params(300.0, 60.0);
    p.initial_depth = 0.3;
    p.rain = Hyetograph::from_mm_h(&[[0.0, 50.0], [1e9, 50.0]]);
    let terr = PreparedTerrain::new(b.clone());
    let mut ws = Workspace::<T>::new(b.layout, &MonitorSet::default(), threads, 32);
    let r = Solver::<T>::new(p).run_impl(&mut ws, &terr, None, &mut itr_core::model::NoopObserver).unwrap();
    println!("{:>4} {n}x{n} threads={threads:>2}: {} steps, {:.2e} cell-upd/s ({:.2e}/thread), {:.2}s", T::NAME, r.steps, r.cell_updates as f64 / r.wall_s, r.cell_updates as f64 / r.wall_s / threads as f64, r.wall_s);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() == 4 {
        let (n, t): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
        if a[1] == "f32" { bench::<f32>(n, t) } else { bench::<f64>(n, t) }
        return;
    }
    for &n in &[256usize, 512] {
        bench::<f32>(n, 1);
        bench::<f64>(n, 1);
    }
    bench::<f32>(1024, 1);
    bench::<f32>(1024, 4);
    bench::<f32>(1024, 8);
}
