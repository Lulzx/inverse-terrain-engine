//! Full-output runs and result files (spec §8.3): rasters, ledger CSV, viewer frames,
//! manifest. Outputs never change the trajectory (§9.1.1).

use crate::io::{self, Res};
use crate::load::Loaded;
use itr_core::design::TerrainEdit;
use itr_core::model::{AbortReason, MonitorSet, Observer, RunSummary, SyncView};
use itr_core::objective::{Objective, ObjectiveReport};
use itr_core::raster::Raster;
use itr_core::Real;
use itr_hydro::{PreparedTerrain, Workspace};
use itr_opt::Problem;
use serde_json::{json, Value};
use std::ops::ControlFlow;
use std::path::Path;

/// Observer that captures depth frames (u16 mm + LZ4) and a strided discharge field.
pub struct FrameRecorder {
    pub nx: usize,
    pub ny: usize,
    pub frames: Vec<(f64, Vec<u8>, bool)>,
    pub vel: Vec<Vec<f32>>,
    pub stride: usize,
    buf: Vec<f32>,
    qx: Vec<f32>,
    qy: Vec<f32>,
}

impl FrameRecorder {
    pub fn new(nx: usize, ny: usize) -> Self {
        let n = nx * ny;
        let stride = ((nx.max(ny) as f64) / 48.0).ceil().max(1.0) as usize;
        Self { nx, ny, frames: vec![], vel: vec![], stride, buf: vec![0.0; n], qx: vec![0.0; n], qy: vec![0.0; n] }
    }
}

impl Observer for FrameRecorder {
    fn on_sync(&mut self, v: &SyncView<'_>) -> ControlFlow<AbortReason> {
        v.state.depth(&mut self.buf);
        let (bytes, sat) = io::encode_frame(&self.buf);
        self.frames.push((v.t, bytes, sat));
        v.state.discharge(&mut self.qx, &mut self.qy);
        let mut f = vec![];
        for r in (self.stride / 2..self.ny).step_by(self.stride) {
            for c in (self.stride / 2..self.nx).step_by(self.stride) {
                let k = r * self.nx + c;
                f.push(self.qx[k]);
                f.push(self.qy[k]);
            }
        }
        self.vel.push(f);
        ControlFlow::Continue(())
    }
}

pub struct FullRun {
    pub summary: RunSummary,
    pub frames: FrameRecorder,
    pub report: ObjectiveReport,
    /// Peak depth per fine cell (NaN on walls).
    pub peak: Raster,
    pub tpeak: Raster,
}

/// Objective over the full domain at level 1 (same assets/guards as the search objective).
pub fn full_objective<T: Real>(pb: &Problem<T>) -> Objective {
    let lv = pb.levels.last().unwrap();
    let mut def = lv.objective.def.clone();
    def.monitor_full_domain = true;
    Objective::new(def)
}

pub fn full_run<T: Real>(pb: &Problem<T>, obj: &Objective, edit: &TerrainEdit, baseline_max: Option<&[f64]>, threads: usize) -> Res<FullRun> {
    let lv = pb.levels.last().unwrap();
    let base = lv.base.clone();
    let mut terr = PreparedTerrain::new(base.clone());
    terr.apply_edit(edit);
    let mon = MonitorSet::all(base.nx(), base.ny());
    let mut ws = Workspace::<T>::new(base.layout, &mon, threads, 16);
    let mut rec = FrameRecorder::new(base.nx(), base.ny());
    let s = pb.solver().run_impl(&mut ws, &terr, None, &mut rec).map_err(|e| e.0)?;
    let report = obj.evaluate(&s.monitor_max, edit, baseline_max);
    let geo = base.geo.clone();
    let mut peak = Raster::new(base.nx(), base.ny(), geo.clone(), 0.0);
    let mut tpeak = Raster::new(base.nx(), base.ny(), geo, 0.0);
    for k in 0..peak.data.len() {
        let wall = base.wall.data[k];
        peak.data[k] = if wall { f64::NAN } else { s.monitor_max[k] };
        tpeak.data[k] = if wall || s.monitor_max[k] <= 0.0 { f64::NAN } else { s.monitor_tmax[k] };
    }
    Ok(FullRun { summary: s, frames: rec, report, peak, tpeak })
}

pub fn write_ledger_csv(path: &Path, s: &RunSummary) -> Res<()> {
    let mut out = String::from("t_s,volume_m3,rain_m3,inflow_m3,outflow_m3,infiltration_m3,positivity_fix_m3,residual_m3\n");
    for r in &s.ledger.rows {
        out += &format!(
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}\n",
            r.t, r.volume_m3, r.rain_m3, r.inflow_m3, r.outflow_m3, r.infiltration_m3, r.positivity_fix_m3, r.residual_m3
        );
    }
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Write frames for the viewer: `frames/<tag>_NNNN.bin` and `frames/<tag>.json`.
pub fn write_frames(dir: &Path, tag: &str, fr: &FrameRecorder) -> Res<Value> {
    let fd = dir.join("frames");
    std::fs::create_dir_all(&fd).map_err(|e| e.to_string())?;
    let mut list = vec![];
    for (k, (t, bytes, sat)) in fr.frames.iter().enumerate() {
        let name = format!("{tag}_{k:04}.bin");
        std::fs::write(fd.join(&name), bytes).map_err(|e| e.to_string())?;
        list.push(json!({"t": t, "file": name, "saturated": sat}));
    }
    let vel: Vec<u8> = fr.vel.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    let vname = format!("{tag}_vel.bin");
    std::fs::write(fd.join(&vname), lz4_flex::block::compress_prepend_size(&vel)).map_err(|e| e.to_string())?;
    Ok(json!({"frames": list, "velocity": {"file": vname, "stride": fr.stride,
        "nx": fr.nx.saturating_sub(fr.stride / 2).div_ceil(fr.stride), "ny": fr.ny.saturating_sub(fr.stride / 2).div_ceil(fr.stride)},
        "encoding": "u16 little-endian millimetres, saturating at 65.535 m; LZ4 block with u32 size prefix"}))
}

pub fn manifest<T: Real>(ld: &Loaded, pb: &Problem<T>, extra: Value) -> Value {
    let sc = &ld.inputs.scenario;
    json!({
        "engine": {"name": "itr", "version": env!("CARGO_PKG_VERSION"), "model_version": itr_core::MODEL_VERSION},
        "scenario_id": sc.scenario_id,
        "scenario_path": ld.scenario_path.display().to_string(),
        "inputs": ld.file_hashes.iter().map(|(p, h)| json!({"path": p, "blake3": h})).collect::<Vec<_>>(),
        "problem_hash": pb.problem_hash,
        "crs": sc.terrain.crs,
        "vertical_datum": sc.terrain.vertical_datum,
        "units": {"length": "m", "time": "s", "depth": "m", "volume": "m3", "rain_input": "mm/h"},
        "z_ref_m": pb.fine.z_ref,
        "grid": {"nx": pb.fine.nx, "ny": pb.fine.ny, "dx": pb.fine.geo.dx, "dy": pb.fine.geo.dy,
                 "origin_x": pb.fine.geo.origin_x, "origin_y": pb.fine.geo.origin_y},
        "precision": T::NAME,
        "solver": sc.solver,
        "determinism": "D0 (bitwise across thread counts/schedules on one build); D1 across x86-64/aarch64 intended (no FMA/libm in kernel), not yet CI-verified",
        "limitations": [
            "first-order finite volume (HLL + hydrostatic reconstruction); no subgrid obstructions or culverts",
            "transmissive boundaries are partially reflective for non-normal or subcritical flow",
            "uniform rainfall; constant-capacity infiltration",
            "results are model outputs on the supplied DEM, not engineering design approval",
        ],
        "run": extra,
    })
}

/// Peak-depth difference raster, and summary of newly worsened / improved areas.
pub fn delta(a: &Raster, b: &Raster, area: f64) -> (Raster, Value) {
    let mut d = a.clone();
    let (mut worse, mut better, mut wa, mut ba) = (0.0f64, 0.0f64, 0usize, 0usize);
    for k in 0..d.data.len() {
        d.data[k] = b.data[k] - a.data[k];
        if d.data[k].is_finite() {
            if d.data[k] > 0.01 {
                wa += 1;
            }
            if d.data[k] < -0.01 {
                ba += 1;
            }
            worse = worse.max(d.data[k]);
            better = better.min(d.data[k]);
        }
    }
    (d, json!({"max_increase_m": worse, "max_decrease_m": -better,
               "area_worsened_gt_1cm_m2": wa as f64 * area, "area_improved_gt_1cm_m2": ba as f64 * area}))
}
