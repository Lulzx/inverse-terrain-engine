//! Candidate evaluation (spec §6.4, §7.5, §7.6): decode → static checks → materialize →
//! warm-started simulation with exact guard / incumbent early abort → objective.
//! Candidates run in parallel, one workspace per worker; results are independent of
//! scheduling (D0).

use crate::problem::Problem;
use crate::search::Fitness;
use itr_core::design::{ConstraintViolation, Primitive, TerrainEdit};
use itr_core::model::{AbortReason, Observer, SyncView};
use itr_core::objective::{Objective, ObjectiveReport};
use itr_core::Real;
use itr_hydro::{PreparedTerrain, Workspace};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    StaticInfeasible,
    GuardAbort,
    IncumbentAbort,
    Unhealthy,
}

/// One evaluation, as written to `search.jsonl` / `cache.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalRecord {
    pub key: String,
    pub level: usize,
    pub status: Status,
    pub primitives: Vec<Primitive>,
    pub fitness_feasible: bool,
    pub j: f64,
    pub violation: f64,
    pub breach_sync: u32,
    pub report: Option<ObjectiveReport>,
    pub static_violations: Vec<ConstraintViolation>,
    pub abort: Option<AbortReason>,
    pub steps: u64,
    pub warm_start_steps: u64,
    pub cell_updates: u64,
    pub wall_s: f64,
    /// Primitives narrower than two cells at this level (flagged, spec §6.5).
    pub unresolved: Vec<usize>,
    /// Subdomain replay outcome: "used", or "escalated: <reason>" (§7.11.3).
    #[serde(default)]
    pub replay: Option<String>,
    /// Locked-Δt fallback time, if the hard-CFL check forced an adaptive restart.
    #[serde(default)]
    pub dt_fallback_t: Option<f64>,
    /// Robust objective (§6.6): per-member results; `j` is then E[J] + β·CVaR_α(J).
    #[serde(default)]
    pub members: Vec<MemberResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemberResult {
    pub name: String,
    pub j: f64,
    pub feasible: bool,
    pub violation: f64,
    pub status: Status,
}

/// E[J] + β·CVaR_α(J) with equal member weights; CVaR_α = mean of the worst
/// ⌈(1−α)·S⌉ members.
pub fn robust_j(js: &[f64], alpha: f64, beta: f64) -> f64 {
    let s = js.len().max(1);
    let mean = js.iter().sum::<f64>() / s as f64;
    let mut v = js.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    let k = (((1.0 - alpha) * s as f64).ceil() as usize).clamp(1, s);
    let cvar = v[..k].iter().sum::<f64>() / k as f64;
    mean + beta * cvar
}

impl EvalRecord {
    pub fn fitness(&self) -> Fitness {
        Fitness { feasible: self.fitness_feasible, j: self.j, violation: self.violation, breach_sync: self.breach_sync }
    }
}

struct GuardObserver<'a> {
    obj: &'a Objective,
    baseline_max: &'a [f64],
    edit: &'a TerrainEdit,
    incumbent: Option<f64>,
}

impl Observer for GuardObserver<'_> {
    fn on_sync(&mut self, v: &SyncView<'_>) -> ControlFlow<AbortReason> {
        if let Some((cell, ex)) = self.obj.guard_breach(v, self.baseline_max) {
            return ControlFlow::Break(AbortReason::GuardViolation { cell, exceedance_m: ex, sync_index: v.sync_index });
        }
        if let Some(inc) = self.incumbent {
            let lb = self.obj.running_lower_bound(v.monitor_max, self.edit);
            if lb >= inc {
                return ControlFlow::Break(AbortReason::Incumbent { lower_bound: lb });
            }
        }
        ControlFlow::Continue(())
    }
}

struct Slot<T: Real> {
    ws: Workspace<T>,
    terr: PreparedTerrain<T>,
    /// ROI workspace when subdomain replay is enabled at this level.
    sub: Option<Workspace<T>>,
}

/// Full-length run of one candidate on a batch backend (no early abort).
#[derive(Clone, Debug)]
pub struct BatchRun {
    pub monitor_max: Vec<f64>,
    pub steps: u64,
    pub cell_updates: u64,
    /// Non-finite or negative depth detected.
    pub bad: bool,
}

/// Population-parallel batch backend (spec §7.8), e.g. the wgpu solver in `itr-gpu`.
/// Implemented outside this crate so the optimizer never depends on a GPU stack.
/// Every candidate runs to the end (no early abort), so traces differ from the CPU
/// backend's within the D3 tolerance; on one adapter they are run-to-run bitwise (D2).
pub trait BatchBackend<T: Real>: Send {
    fn name(&self) -> String;
    /// `Err` means "unsupported here" (the evaluator then uses the CPU for this level).
    fn run(&mut self, p: &itr_hydro::SolverParams, terrains: &[&PreparedTerrain<T>], monitors: &itr_core::model::MonitorSet) -> Result<Vec<BatchRun>, String>;
}

/// Per-level evaluator: worker slots and the evaluation cache.
pub struct Evaluator<T: Real> {
    pool: rayon::ThreadPool,
    /// Free list of preallocated workspaces per level. A candidate checks one out for
    /// the duration of its run. (Indexing by worker thread is unsound here: a worker
    /// blocked in a nested grid-parallel `install` may steal another candidate.)
    slots: Vec<Mutex<Vec<Slot<T>>>>,
    hint: Arc<dyn Fn() -> usize + Send + Sync>,
    pub cache: HashMap<String, EvalRecord>,
    pub sims: usize,
    pub cell_updates: u64,
    pub cache_hits: usize,
    workers: usize,
    /// Candidates currently simulating (drives tail re-partitioning, §7.11.5).
    running: Arc<AtomicUsize>,
    /// Utilization accounting: Σ candidate wall time and Σ batch wall time × workers.
    pub busy_s: f64,
    pub capacity_s: f64,
    /// Optional batch backend and, per level, whether it was rejected there.
    batch: Option<Box<dyn BatchBackend<T>>>,
    batch_off: Vec<bool>,
    /// Simulations run on the batch backend.
    pub batch_sims: usize,
}

/// Prepared (decoded + materialized + statically checked) candidate.
pub struct Prepared {
    pub key: String,
    pub prims: Vec<Primitive>,
    pub edit: TerrainEdit,
    pub violations: Vec<ConstraintViolation>,
}

impl Prepared {
    pub fn empty() -> Self {
        Self { key: String::new(), prims: vec![], edit: TerrainEdit::default(), violations: vec![] }
    }
}

impl<T: Real> Evaluator<T> {
    pub fn new(pb: &Problem<T>, workers: usize) -> Self {
        let workers = workers.max(1);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().expect("pool");
        let running = Arc::new(AtomicUsize::new(0));
        // Each workspace asks, at every sync point, for its share of idle cores (§7.11.5).
        let hint: Arc<dyn Fn() -> usize + Send + Sync> = {
            let r = running.clone();
            Arc::new(move || (workers / r.load(Ordering::Relaxed).max(1)).max(1))
        };
        let slots = (0..pb.levels.len()).map(|li| Mutex::new((0..workers).map(|_| make_slot(pb, li, &hint)).collect())).collect();
        Self {
            pool,
            slots,
            hint,
            cache: HashMap::new(),
            sims: 0,
            cell_updates: 0,
            cache_hits: 0,
            workers,
            running,
            busy_s: 0.0,
            capacity_s: 0.0,
            batch: None,
            batch_off: vec![false; pb.levels.len()],
            batch_sims: 0,
        }
    }

    /// Route plain evaluations (HLL physics, no subdomain replay, no ensemble) through a
    /// batch backend. Levels it rejects fall back to the CPU, with a note on stderr.
    pub fn with_batch_backend(mut self, b: Box<dyn BatchBackend<T>>) -> Self {
        self.batch = Some(b);
        self
    }

    fn batch_eligible(&self, pb: &Problem<T>, li: usize) -> bool {
        let lv = &pb.levels[li];
        self.batch.is_some() && !self.batch_off[li] && lv.replay.is_none() && lv.extra.is_empty() && pb.levels[li].params.kernel == itr_hydro::KernelKind::Fused
    }

    /// Run `idx` (simulating candidates) on the batch backend; `None` if it declined.
    fn eval_on_batch(&mut self, pb: &Problem<T>, li: usize, cands: &[Prepared], keys: &[String], idx: &[usize]) -> Option<Vec<EvalRecord>> {
        let lv = &pb.levels[li];
        let terrs: Vec<PreparedTerrain<T>> = idx
            .iter()
            .map(|&k| {
                let mut t = PreparedTerrain::new(lv.base.clone());
                t.apply_edit(&cands[k].edit);
                t
            })
            .collect();
        let refs: Vec<&PreparedTerrain<T>> = terrs.iter().collect();
        let t0 = std::time::Instant::now();
        let b = self.batch.as_mut().unwrap();
        let runs = match b.run(&lv.params, &refs, &lv.objective.monitors) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("note: {} backend declined level {} ({e}); using the CPU there", b.name(), lv.level);
                self.batch_off[li] = true;
                return None;
            }
        };
        let wall = t0.elapsed().as_secs_f64() / idx.len().max(1) as f64;
        Some(
            idx.iter()
                .zip(runs)
                .map(|(&k, r)| {
                    let c = &cands[k];
                    let mut rec = new_record(pb, li, c, keys[k].clone());
                    rec.steps = r.steps;
                    rec.cell_updates = r.cell_updates;
                    rec.wall_s = wall;
                    if r.bad {
                        rec.status = Status::Unhealthy;
                        rec.abort = Some(AbortReason::Unhealthy { message: "batch backend: non-finite or negative depth".into() });
                        rec.fitness_feasible = false;
                        rec.violation = 1e9;
                        rec.j = 1e300;
                        return rec;
                    }
                    let report = lv.objective.evaluate(&r.monitor_max, &c.edit, Some(&lv.baseline_max));
                    rec.j = report.j;
                    rec.fitness_feasible = report.feasible;
                    rec.violation = report.guard_violation_m;
                    rec.report = Some(report);
                    rec
                })
                .collect(),
        )
    }

    /// Achieved candidate-parallel utilization U = Σ run time / (Σ batch time × workers)
    /// (a lower bound: threads lent to tail candidates are not counted).
    pub fn utilization(&self) -> f64 {
        if self.capacity_s > 0.0 { (self.busy_s / self.capacity_s).min(1.0) } else { 1.0 }
    }

    /// Decode and statically check `u` (no simulation).
    pub fn prepare(pb: &Problem<T>, li: usize, u: &[f64]) -> Prepared {
        Self::prepare_in(pb, li, u, false)
    }

    /// Decode in the free (`corridor = false`) or corridor-native design space. The cache
    /// key depends only on the decoded, quantized primitives, so both spaces share it.
    pub fn prepare_in(pb: &Problem<T>, li: usize, u: &[f64], corridor: bool) -> Prepared {
        let ds = pb.design.as_ref().expect("design space");
        let prims = match (&pb.corridor, corridor) {
            (Some(cs), true) => cs.decode(u),
            _ => ds.decode(u),
        };
        Self::prepare_prims(pb, li, prims)
    }

    /// Prepare explicit (already decoded) primitives, e.g. ablation subsets.
    pub fn prepare_prims(pb: &Problem<T>, li: usize, prims: Vec<Primitive>) -> Prepared {
        let ds = pb.design.as_ref().expect("design space");
        let d = &pb.inputs.dem;
        let edit = ds.materialize(&prims, &d.geo, d.nx, d.ny, &pb.editable);
        let mut violations = vec![];
        ds.static_violations(&edit, d, &mut violations);
        let mut h = itr_core::hash::Hasher::default();
        h.str(&pb.problem_hash).u64(pb.levels[li].level as u64).i64s(&ds.cache_key(&prims));
        Prepared { key: h.hex(), prims, edit, violations }
    }

    pub fn cache_key(c: &Prepared, incumbent: Option<f64>) -> String {
        match incumbent {
            Some(i) => format!("{}:{:016x}", c.key, i.to_bits()),
            None => c.key.clone(),
        }
    }

    /// Evaluate a batch at level index `li`. `incumbent` (elitist methods only) is frozen
    /// for the whole batch and is part of the cache key so replay is exact.
    pub fn eval_batch(&mut self, pb: &Problem<T>, li: usize, cands: &[Prepared], incumbent: Option<f64>) -> Vec<EvalRecord> {
        let keys: Vec<String> = cands.iter().map(|c| Self::cache_key(c, incumbent)).collect();
        let todo: Vec<usize> = (0..cands.len()).filter(|&k| !self.cache.contains_key(&keys[k])).collect();
        // Duplicates inside the batch are evaluated once.
        let mut uniq: Vec<usize> = vec![];
        for &k in &todo {
            if !uniq.iter().any(|&u| keys[u] == keys[k]) {
                uniq.push(k);
            }
        }
        let n_uniq = uniq.len();
        let mut fresh_batch = vec![];
        if self.batch_eligible(pb, li) {
            let (sim, stat): (Vec<usize>, Vec<usize>) = uniq.iter().partition(|&&k| cands[k].violations.is_empty());
            if !sim.is_empty()
                && let Some(recs) = self.eval_on_batch(pb, li, cands, &keys, &sim)
            {
                self.batch_sims += recs.len();
                fresh_batch = recs;
                uniq = stat;
            }
        }
        let slots = &self.slots[li];
        let running = &self.running;
        let hint = &self.hint;
        let t0 = std::time::Instant::now();
        let fresh: Vec<EvalRecord> = self.pool.install(|| {
            uniq.par_iter()
                .map(|&k| {
                    let taken = slots.lock().unwrap().pop();
                    let mut slot = taken.unwrap_or_else(|| make_slot(pb, li, hint));
                    if slot.ws.threads > 1 {
                        slot.ws.set_threads(1, 32);
                    }
                    running.fetch_add(1, Ordering::Relaxed);
                    let r = eval_one(pb, li, &cands[k], keys[k].clone(), incumbent, &mut slot);
                    running.fetch_sub(1, Ordering::Relaxed);
                    slots.lock().unwrap().push(slot);
                    r
                })
                .collect()
        });
        // U counts CPU simulations only (batch-backend runs are reported separately).
        if fresh.iter().any(|r| r.status != Status::StaticInfeasible) {
            self.capacity_s += t0.elapsed().as_secs_f64() * self.workers as f64;
            self.busy_s += fresh.iter().map(|r| r.wall_s).sum::<f64>();
        }
        for r in fresh.into_iter().chain(fresh_batch) {
            if r.status != Status::StaticInfeasible {
                self.sims += 1;
            }
            self.cell_updates += r.cell_updates;
            self.cache.insert(r.key.clone(), r);
        }
        self.cache_hits += cands.len() - n_uniq;
        keys.iter().map(|k| self.cache[k].clone()).collect()
    }
}

fn make_slot<T: Real>(pb: &Problem<T>, li: usize, hint: &Arc<dyn Fn() -> usize + Send + Sync>) -> Slot<T> {
    let lv = &pb.levels[li];
    let sub = lv.replay.as_ref().map(|r| {
        let mut w = Workspace::new(r.tape.sub_layout, &r.monitors, 1, 32);
        w.replay = Some((r.tape.clone(), r.tol_h, r.tol_q));
        w.threads_hint = Some(hint.clone());
        w
    });
    let mut ws = Workspace::new(lv.base.layout, &lv.objective.monitors, 1, 32);
    ws.threads_hint = Some(hint.clone());
    Slot { ws, terr: PreparedTerrain::new(lv.base.clone()), sub }
}

fn new_record<T: Real>(pb: &Problem<T>, li: usize, c: &Prepared, key: String) -> EvalRecord {
    let lv = &pb.levels[li];
    let cell = lv.base.dx.min(lv.base.dy);
    let unresolved = c.prims.iter().enumerate().filter(|(_, p)| p.height != 0.0 && p.width < 2.0 * cell).map(|(i, _)| i).collect();
    EvalRecord {
        key,
        level: lv.level,
        status: Status::Ok,
        primitives: c.prims.clone(),
        fitness_feasible: true,
        j: 0.0,
        violation: 0.0,
        breach_sync: 0,
        report: None,
        static_violations: c.violations.clone(),
        abort: None,
        steps: 0,
        warm_start_steps: 0,
        cell_updates: 0,
        wall_s: 0.0,
        unresolved,
        replay: None,
        dt_fallback_t: None,
        members: vec![],
    }
}

fn eval_one<T: Real>(pb: &Problem<T>, li: usize, c: &Prepared, key: String, incumbent: Option<f64>, slot: &mut Slot<T>) -> EvalRecord {
    let lv = &pb.levels[li];
    let mut rec = new_record(pb, li, c, key);
    if !c.violations.is_empty() {
        rec.status = Status::StaticInfeasible;
        rec.fitness_feasible = false;
        // Static violations rank behind every simulated candidate.
        rec.violation = 1e6 + c.violations.iter().map(|v| v.amount).sum::<f64>();
        rec.j = 1e300;
        return rec;
    }
    let Slot { ws, terr, sub } = slot;
    terr.apply_edit(&c.edit);
    let start = lv.record.best_for(terr.edit_rows);
    let mut obs = GuardObserver { obj: &lv.objective, baseline_max: &lv.baseline_max, edit: &c.edit, incumbent };
    let solver = pb.solver_at(li);
    let mut res = None;
    if let (Some(r), Some(sw)) = (&lv.replay, sub.as_mut()) {
        // Subdomain replay on the ROI (cold start, locked Δt); escalate if invalid.
        let st = itr_hydro::replay::sub_terrain(terr, r.roi);
        match solver.run_impl(sw, &st, None, &mut obs) {
            Ok(s) => match &s.aborted {
                Some(AbortReason::ReplayInvalid { reason, .. }) => rec.replay = Some(format!("escalated: {reason}")),
                _ => {
                    rec.replay = Some("used".into());
                    res = Some(Ok(s));
                }
            },
            Err(e) => rec.replay = Some(format!("escalated: {}", e.0)),
        }
        obs = GuardObserver { obj: &lv.objective, baseline_max: &lv.baseline_max, edit: &c.edit, incumbent };
    }
    let res = match res {
        Some(r) => r,
        None => solver.run_impl(ws, terr, if c.edit.is_empty() { None } else { start }, &mut obs),
    };
    let s = match res {
        Ok(s) => s,
        Err(e) => {
            rec.status = Status::Unhealthy;
            rec.abort = Some(AbortReason::Unhealthy { message: e.0 });
            rec.fitness_feasible = false;
            rec.violation = 1e9;
            rec.j = 1e300;
            return rec;
        }
    };
    rec.steps = s.steps;
    rec.dt_fallback_t = s.dt_fallback_t;
    rec.warm_start_steps = s.warm_start_steps;
    rec.cell_updates = s.cell_updates;
    rec.wall_s = s.wall_s;
    let report = lv.objective.evaluate(&s.monitor_max, &c.edit, Some(&lv.baseline_max));
    rec.j = report.j;
    match &s.aborted {
        None => {
            rec.fitness_feasible = report.feasible;
            rec.violation = report.guard_violation_m;
        }
        Some(AbortReason::GuardViolation { exceedance_m, sync_index, .. }) => {
            rec.status = Status::GuardAbort;
            rec.fitness_feasible = false;
            rec.violation = *exceedance_m;
            rec.breach_sync = *sync_index;
        }
        Some(AbortReason::Incumbent { lower_bound }) => {
            rec.status = Status::IncumbentAbort;
            rec.j = *lower_bound;
        }
        Some(AbortReason::Unhealthy { .. }) | Some(AbortReason::ReplayInvalid { .. }) => {
            rec.status = Status::Unhealthy;
            rec.fitness_feasible = false;
            rec.violation = 1e9;
            rec.j = 1e300;
        }
    }
    rec.abort = s.aborted.clone();
    rec.report = Some(report);
    if !lv.extra.is_empty() {
        robust_members(pb, li, c, &mut rec, ws, terr);
    }
    rec
}

/// Evaluate the remaining ensemble members and fold them into a robust record.
fn robust_members<T: Real>(pb: &Problem<T>, li: usize, c: &Prepared, rec: &mut EvalRecord, ws: &mut Workspace<T>, terr: &PreparedTerrain<T>) {
    let lv = &pb.levels[li];
    let rc = pb.inputs.scenario.robustness.as_ref().expect("robustness config");
    let first = MemberResult {
        name: rc.members[0].name.clone(),
        j: rec.j,
        feasible: rec.fitness_feasible,
        violation: rec.violation,
        status: rec.status,
    };
    rec.members.push(first);
    if rec.status == Status::Ok {
        for m in &lv.extra {
            let start = m.record.best_for(terr.edit_rows);
            let mut obs = GuardObserver { obj: &lv.objective, baseline_max: &m.baseline_max, edit: &c.edit, incumbent: None };
            let solver = itr_hydro::Solver::<T>::new(m.params.clone());
            let r = match solver.run_impl(ws, terr, if c.edit.is_empty() { None } else { start }, &mut obs) {
                Ok(s) => {
                    rec.steps += s.steps;
                    rec.warm_start_steps += s.warm_start_steps;
                    rec.cell_updates += s.cell_updates;
                    rec.wall_s += s.wall_s;
                    let rep = lv.objective.evaluate(&s.monitor_max, &c.edit, Some(&m.baseline_max));
                    match &s.aborted {
                        None => MemberResult { name: m.name.clone(), j: rep.j, feasible: rep.feasible, violation: rep.guard_violation_m, status: Status::Ok },
                        Some(AbortReason::GuardViolation { exceedance_m, .. }) => {
                            MemberResult { name: m.name.clone(), j: rep.j, feasible: false, violation: *exceedance_m, status: Status::GuardAbort }
                        }
                        Some(_) => MemberResult { name: m.name.clone(), j: 1e300, feasible: false, violation: 1e9, status: Status::Unhealthy },
                    }
                }
                Err(_) => MemberResult { name: m.name.clone(), j: 1e300, feasible: false, violation: 1e9, status: Status::Unhealthy },
            };
            let stop = !r.feasible;
            rec.members.push(r);
            if stop {
                break;
            }
        }
    }
    let feasible = rec.members.iter().all(|m| m.feasible) && rec.members.len() == lv.extra.len() + 1;
    let js: Vec<f64> = rec.members.iter().map(|m| m.j).collect();
    rec.j = robust_j(&js, rc.alpha, rc.beta);
    rec.fitness_feasible = feasible;
    if !feasible {
        rec.violation = rec.members.iter().map(|m| m.violation).fold(0.0, f64::max).max(1e-12);
        if let Some(m) = rec.members.iter().find(|m| m.status == Status::GuardAbort)
            && rec.status == Status::Ok {
                rec.status = m.status;
            }
    }
}
