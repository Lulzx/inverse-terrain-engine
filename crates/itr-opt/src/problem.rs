//! Problem assembly (spec §4, §6): loaded inputs → per-fidelity-level terrain, monitors,
//! objective and baseline. No file I/O here; the CLI loads rasters and masks.

use itr_core::design::{DesignSpace, PrimitiveSpec};
use itr_core::model::NoopObserver;
use itr_core::objective::{Asset, Objective, ObjectiveDef};
use itr_core::raster::{Mask, Raster};
use itr_core::scenario::Scenario;
use itr_core::Real;
use itr_core::model::MonitorSet;
use itr_core::scenario::DtMode;
use itr_hydro::replay::{Roi, Tape};
use itr_hydro::{BaselineRecord, FineTerrain, PreparedTerrain, Solver, SolverParams, TerrainBase, Workspace};
use std::sync::Arc;

/// Everything the engine needs, already loaded and on one grid.
#[derive(Clone)]
pub struct ProblemInputs {
    pub scenario: Scenario,
    pub dem: Raster,
    /// nodata ∪ obstacles.
    pub walls: Mask,
    pub manning: Vec<f64>,
    pub editable: Option<Mask>,
    /// Named protected-asset masks with optional per-asset threshold/weight.
    pub assets: Vec<(String, Mask, Option<f64>, f64)>,
    pub guard: Option<Mask>,
    pub primitives: Vec<PrimitiveSpec>,
}

/// Baseline state and objective at one fidelity level.
pub struct Level<T: Real> {
    pub level: usize,
    pub base: Arc<TerrainBase<T>>,
    pub objective: Objective,
    pub baseline_max: Vec<f64>,
    pub baseline: itr_core::model::RunSummary,
    pub record: BaselineRecord<T>,
    /// Candidate solver parameters at this level (locked Δt schedule when enabled).
    pub params: SolverParams,
    pub replay: Option<LevelReplay<T>>,
    /// Additional ensemble members (robust objective, §6.6); member 0 is this level itself.
    pub extra: Vec<Member<T>>,
}

/// One ensemble member at one level: its forcing, baseline and warm-start record.
pub struct Member<T: Real> {
    pub name: String,
    pub params: SolverParams,
    pub baseline_max: Vec<f64>,
    pub baseline: itr_core::model::RunSummary,
    pub record: BaselineRecord<T>,
}

/// Scale forcing for an ensemble member.
pub fn scaled(p: &SolverParams, rain: f64, inflow: f64) -> SolverParams {
    let mut q = p.clone();
    for pt in q.rain.pts.iter_mut() {
        pt.1 *= rain;
    }
    q.inflow_scale *= inflow;
    q
}

/// Subdomain replay set-up for one level (§7.11.3).
pub struct LevelReplay<T: Real> {
    pub tape: Arc<Tape<T>>,
    pub roi: Roi,
    /// Objective monitor cells mapped into the ROI grid (same order).
    pub monitors: MonitorSet,
    pub tol_h: f64,
    pub tol_q: f64,
}

pub struct Problem<T: Real> {
    pub inputs: ProblemInputs,
    pub fine: Arc<FineTerrain>,
    pub design: Option<DesignSpace>,
    pub editable: Mask,
    pub params: SolverParams,
    pub levels: Vec<Level<T>>,
    /// Problem hash (BLAKE3 of canonical problem JSON + input data + model version).
    pub problem_hash: String,
    /// Corridor-native design space (`design.placement = "corridor"`, §7.11.4).
    pub corridor: Option<crate::corridor::CorridorSpace>,
}

/// Map a fine-grid mask to coarse cell indices (a coarse cell is selected if any of its
/// fine cells is).
pub fn coarse_cells(m: &Mask, lv: usize) -> Vec<u32> {
    let nx = m.nx.div_ceil(lv);
    let mut v: Vec<u32> = m.indices().iter().map(|&k| {
        let (c, r) = (k as usize % m.nx, k as usize / m.nx);
        ((r / lv) * nx + c / lv) as u32
    }).collect();
    v.sort_unstable();
    v.dedup();
    v
}

impl<T: Real> Problem<T> {
    /// Build all levels and run their baselines (with checkpoint recording for warm start).
    pub fn build(mut inputs: ProblemInputs, threads: usize, checkpoint_budget: usize) -> Result<Self, String> {
        let sc = inputs.scenario.clone();
        let (nx, ny) = (inputs.dem.nx, inputs.dem.ny);
        let obj = sc.objectives.clone();
        let wall_mode = obj.as_ref().is_some_and(|o| o.building_mode == "wall");
        // Wall mode: buildings become obstacles; their outer ring is monitored instead.
        let mut asset_masks: Vec<(String, Mask, f64, f64)> = vec![];
        for (name, m, thr, w) in &inputs.assets {
            let thr = thr.or(obj.as_ref().map(|o| o.depth_threshold_m)).unwrap_or(0.1);
            if wall_mode {
                inputs.walls.or(m);
                asset_masks.push((name.clone(), m.outer_ring(), thr, *w));
            } else {
                asset_masks.push((name.clone(), m.clone(), thr, *w));
            }
        }
        let fine = Arc::new(FineTerrain::new(&inputs.dem, &inputs.walls, &inputs.manning));
        let editable = match &inputs.editable {
            Some(e) => {
                let mut e = e.clone();
                for (i, v) in e.data.iter_mut().enumerate() {
                    *v = *v && !fine.wall.data[i];
                }
                e
            }
            None => Mask::new(nx, ny, false),
        };
        let design = match &sc.design {
            Some(d) if !inputs.primitives.is_empty() => Some(DesignSpace::new(inputs.primitives.clone(), &editable, &inputs.dem.geo, d)?),
            _ => None,
        };
        // Guard cells: explicit mask, or the whole map minus protected cells.
        let mut guard = inputs.guard.clone().unwrap_or_else(|| Mask::new(nx, ny, false));
        if obj.as_ref().is_some_and(|o| o.guard_whole_map) {
            let mut prot = Mask::new(nx, ny, false);
            for (_, m, _, _) in &inputs.assets {
                prot.or(m);
            }
            for i in 0..guard.data.len() {
                guard.data[i] = !prot.data[i] && !fine.wall.data[i];
            }
        }
        let params = SolverParams::from_scenario(&sc);
        let levels_cfg = sc.optimizer.fidelity_levels.clone();
        let mut levels = vec![];
        let mut q_int: Option<(Vec<f64>, Vec<f64>)> = None;
        for &lv in &levels_cfg {
            let base = Arc::new(TerrainBase::<T>::build(fine.clone(), lv, &sc.boundaries.segments));
            let (cnx, cny) = (base.nx(), base.ny());
            let def = ObjectiveDef {
                assets: asset_masks
                    .iter()
                    .map(|(n, m, thr, w)| Asset { name: n.clone(), cells: coarse_cells(m, lv), threshold_m: *thr, weight: *w })
                    .collect(),
                guard_cells: coarse_cells(&guard, lv),
                guard_tolerance_m: obj.as_ref().map_or(0.02, |o| o.guard_tolerance_m),
                tau_m: obj.as_ref().map_or(0.02, |o| o.smoothing_m),
                cut_cost: sc.design.as_ref().map_or(1.0, |d| d.cut_cost_per_m3),
                fill_cost: sc.design.as_ref().map_or(1.0, |d| d.fill_cost_per_m3),
                earth_weight: obj.as_ref().map_or(0.0, |o| o.earthwork_weight),
                require_no_offsite_worsening: sc.design.as_ref().is_some_and(|d| d.require_no_offsite_worsening),
                monitor_full_domain: obj.as_ref().is_some_and(|o| o.monitor_full_domain),
                nx: cnx,
                ny: cny,
            };
            let objective = Objective::new(def);
            let terr = PreparedTerrain::new(base.clone());
            let mut ws = Workspace::<T>::new(base.layout, &objective.monitors, threads, 32);
            ws.record = Some((checkpoint_budget, BaselineRecord::default()));
            let members = sc.robustness.as_ref().map(|r| r.members.clone()).unwrap_or_default();
            let m0 = members.first();
            let mut level_params = match m0 {
                Some(m) => scaled(&params, m.rain_scale, m.inflow_scale),
                None => params.clone(),
            };
            let screening = lv > 1 && sc.solver.screening_physics == "local_inertial";
            if screening {
                level_params.kernel = itr_hydro::KernelKind::LocalInertial;
            }
            let solver = Solver::<T>::new(level_params.clone());
            let baseline = if lv == 1 {
                let mut qa = crate::corridor::QAccum::new(nx * ny, sc.hydrology.sync_interval_s);
                let b = solver.run_impl(&mut ws, &terr, None, &mut qa).map_err(|e| e.0)?;
                q_int = Some((qa.qx, qa.qy));
                b
            } else {
                solver.run_impl(&mut ws, &terr, None, &mut NoopObserver).map_err(|e| e.0)?
            };
            if let Some(a) = &baseline.aborted {
                return Err(format!("baseline run at level {lv} aborted: {a:?}"));
            }
            let record = ws.take_record().unwrap_or_default();
            let mut lp = level_params.clone();
            if sc.solver.dt_mode == DtMode::BaselineLocked {
                lp.locked_dt = Some(Arc::new(record.dts.clone()));
            }
            // Replay rings carry cell-centred momentum; not used with staggered screening physics.
            let replay = if sc.solver.subdomain_replay == "auto" && !screening {
                choose_roi(&base, &objective, &editable, &asset_masks_all(&inputs), &guard, lv, &baseline, &sc)
                    .and_then(|roi| record_tape::<T>(&base, &objective, &solver, roi, checkpoint_budget.max(256 << 20), threads))
            } else {
                None
            };
            let mut extra = vec![];
            for m in members.iter().skip(1) {
                let mut mp = scaled(&level_params, m.rain_scale / m0.map_or(1.0, |a| a.rain_scale.max(1e-300)), m.inflow_scale / m0.map_or(1.0, |a| a.inflow_scale.max(1e-300)));
                let mut ws = Workspace::<T>::new(base.layout, &objective.monitors, threads, 32);
                ws.record = Some((checkpoint_budget / members.len().max(1), BaselineRecord::default()));
                let mb = Solver::<T>::new(mp.clone()).run_impl(&mut ws, &terr, None, &mut NoopObserver).map_err(|e| e.0)?;
                if let Some(a) = &mb.aborted {
                    return Err(format!("baseline of ensemble member {} at level {lv} aborted: {a:?}", m.name));
                }
                let rec = ws.take_record().unwrap_or_default();
                if sc.solver.dt_mode == DtMode::BaselineLocked {
                    mp.locked_dt = Some(Arc::new(rec.dts.clone()));
                }
                extra.push(Member { name: m.name.clone(), params: mp, baseline_max: mb.monitor_max.clone(), baseline: mb, record: rec });
            }
            levels.push(Level { level: lv, base, baseline_max: baseline.monitor_max.clone(), objective, baseline, record, params: lp, replay, extra });
        }
        let mut h = itr_core::hash::Hasher::default();
        h.str(itr_core::MODEL_VERSION).str(&sc.problem_json()).str(&sc.solver_json()).str(T::NAME);
        h.f64s(&inputs.dem.data).f64s(&inputs.manning);
        h.u64(inputs.walls.count() as u64).u64(editable.count() as u64);
        let problem_hash = h.hex();
        let corridor = match (&sc.design, &design, &q_int) {
            (Some(d), Some(ds), Some((qx, qy))) if d.placement == "corridor" => {
                let mut targets: Vec<(String, Mask)> = asset_masks.iter().map(|(n, m, _, _)| (n.clone(), m.clone())).collect();
                if guard.count() > 0 && !obj.as_ref().is_some_and(|o| o.guard_whole_map) {
                    targets.push(("guard".into(), guard.clone()));
                }
                let cs = crate::corridor::extract(qx, qy, nx, ny, &inputs.dem.geo, &targets, &editable, 4);
                if cs.is_empty() {
                    None
                } else {
                    Some(crate::corridor::CorridorSpace { corridors: cs, specs: ds.specs.clone(), quant: ds.quant.clone() })
                }
            }
            _ => None,
        };
        Ok(Self { inputs, fine, design, editable, params, levels, problem_hash, corridor })
    }

    pub fn solver(&self) -> Solver<T> {
        Solver::new(self.params.clone())
    }
    /// Solver for candidate runs at level index `li`.
    pub fn solver_at(&self, li: usize) -> Solver<T> {
        Solver::new(self.levels[li].params.clone())
    }
}

fn asset_masks_all(inputs: &ProblemInputs) -> Mask {
    let mut m = Mask::new(inputs.dem.nx, inputs.dem.ny, false);
    for (_, a, _, _) in &inputs.assets {
        m.or(a);
    }
    m
}

/// ROI = bbox of editable ∪ protected ∪ guard cells (at this level), dilated by the
/// distance a baseline front travels in two sync intervals (≥ 16 cells). Used only if it
/// covers ≤ 50% of the domain and contains every monitored cell.
#[allow(clippy::too_many_arguments)]
fn choose_roi<T: Real>(
    base: &TerrainBase<T>,
    obj: &Objective,
    editable: &Mask,
    assets: &Mask,
    guard: &Mask,
    lv: usize,
    baseline: &itr_core::model::RunSummary,
    sc: &Scenario,
) -> Option<Roi> {
    let (nx, ny) = (base.nx(), base.ny());
    let mut cells = coarse_cells(editable, lv);
    cells.extend(coarse_cells(assets, lv));
    cells.extend(coarse_cells(guard, lv));
    if cells.is_empty() {
        return None;
    }
    let (mut c0, mut r0, mut c1, mut r1) = (usize::MAX, usize::MAX, 0, 0);
    for &k in &cells {
        let (c, r) = (k as usize % nx, k as usize / nx);
        c0 = c0.min(c);
        r0 = r0.min(r);
        c1 = c1.max(c + 1);
        r1 = r1.max(r + 1);
    }
    let travel = if baseline.dt_min > 0.0 { sc.hydrology.sync_interval_s * sc.solver.cfl / baseline.dt_min } else { 0.0 };
    let m = (travel.ceil() as usize).max(16);
    let roi = Roi { c0: c0.saturating_sub(m), r0: r0.saturating_sub(m), w: 0, h: 0 };
    let roi = Roi { w: (c1 + m).min(nx) - roi.c0, h: (r1 + m).min(ny) - roi.r0, ..roi };
    if 2 * roi.w * roi.h > nx * ny {
        return None;
    }
    if !obj.monitors.cells.iter().all(|&c| roi.contains_cell(c, nx)) {
        return None;
    }
    Some(roi)
}

/// Re-run the baseline (identical trajectory) recording the ROI ring/band tape.
fn record_tape<T: Real>(base: &Arc<TerrainBase<T>>, obj: &Objective, solver: &Solver<T>, roi: Roi, budget: usize, threads: usize) -> Option<LevelReplay<T>> {
    let terr = PreparedTerrain::new(base.clone());
    let mut ws = Workspace::<T>::new(base.layout, &obj.monitors, threads, 32);
    let rec = BaselineRecord { tape: Some(Tape::new(roi, &base.layout, budget)), ..Default::default() };
    ws.record = Some((0, rec));
    solver.run_impl(&mut ws, &terr, None, &mut NoopObserver).ok()?;
    let tape = ws.take_record()?.tape?;
    if tape.overflow {
        return None;
    }
    let nx = base.nx();
    let monitors = MonitorSet { cells: obj.monitors.cells.iter().map(|&c| roi.map_cell(c, nx)).collect() };
    Some(LevelReplay { tape: Arc::new(tape), roi, monitors, tol_h: 2e-3, tol_q: 1e-3 })
}
