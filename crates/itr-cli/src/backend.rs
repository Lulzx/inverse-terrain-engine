//! Evaluator backends (spec §7.8): `solver.backend = "gpu"` routes population batches
//! through the wgpu solver (`itr-gpu`, `f32` only, built with `--features gpu`).

use crate::io::Res;
use itr_core::scenario::Backend;
use itr_core::Real;
use itr_opt::{Evaluator, Problem};

pub trait Backends: Real {
    fn evaluator(pb: &Problem<Self>, workers: usize, backend: Backend) -> Res<Evaluator<Self>>;
}

impl Backends for f64 {
    fn evaluator(pb: &Problem<f64>, workers: usize, backend: Backend) -> Res<Evaluator<f64>> {
        match backend {
            Backend::Cpu => Ok(Evaluator::new(pb, workers)),
            Backend::Gpu => Err("the GPU backend is f32 only; set solver.precision = \"f32\"".into()),
        }
    }
}

impl Backends for f32 {
    fn evaluator(pb: &Problem<f32>, workers: usize, backend: Backend) -> Res<Evaluator<f32>> {
        let ev = Evaluator::new(pb, workers);
        match backend {
            Backend::Cpu => Ok(ev),
            Backend::Gpu => gpu(ev),
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn gpu(_: Evaluator<f32>) -> Res<Evaluator<f32>> {
    Err("this build has no GPU backend; rebuild with `cargo build --release -p itr-cli --features gpu`".into())
}

#[cfg(feature = "gpu")]
fn gpu(ev: Evaluator<f32>) -> Res<Evaluator<f32>> {
    let g = itr_gpu::GpuSolver::new()?;
    eprintln!("gpu backend: {}", g.adapter_info());
    Ok(ev.with_batch_backend(Box::new(Gpu(g))))
}

#[cfg(feature = "gpu")]
struct Gpu(itr_gpu::GpuSolver);

#[cfg(feature = "gpu")]
impl itr_opt::evaluator::BatchBackend<f32> for Gpu {
    fn name(&self) -> String {
        format!("gpu ({})", self.0.adapter_info())
    }
    fn run(&mut self, p: &itr_hydro::SolverParams, terrains: &[&itr_hydro::PreparedTerrain<f32>], monitors: &itr_core::model::MonitorSet) -> Result<Vec<itr_opt::evaluator::BatchRun>, String> {
        if p.inflow_scale != 1.0 {
            return Err("inflow scaling is not supported on the GPU backend".into());
        }
        let cells = terrains.first().map_or(0, |t| (t.base.nx() * t.base.ny()) as u64);
        Ok(self
            .0
            .run_batch(p, terrains, monitors)?
            .into_iter()
            .map(|r| itr_opt::evaluator::BatchRun { cell_updates: r.steps * cells, monitor_max: r.monitor_max, steps: r.steps, bad: r.bad })
            .collect())
    }
}
