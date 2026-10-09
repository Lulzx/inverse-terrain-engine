//! Run loop (spec §5.3, §5.4, §7.5, §7.6): CFL time stepping with sync points, mass
//! ledger, observers, baseline checkpoints and exact warm start.

use crate::aligned::{AlignedBuf, Layout};
use crate::boundary::fill_ghosts;
use crate::kernel::{step_fused, step_oracle, KernelScratch, Monitors, StepCtx};
use crate::numerics::StepConsts;
use crate::terrain::PreparedTerrain;
use itr_core::model::{
    AbortReason, ForwardModel, LedgerReport, LedgerRow, MonitorSet, Observer, RunSummary, SimulationError, StateAccess, SyncView,
};
use itr_core::scenario::{Hyetograph, InfiltrationCfg, Scenario};
use itr_core::{FixedSum, Real};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    Oracle,
    Fused,
    /// Local-inertial screening physics (staggered face discharges, §7.11.2).
    LocalInertial,
}

#[derive(Clone, Debug)]
pub struct SolverParams {
    pub duration_s: f64,
    pub sync_interval_s: f64,
    pub cfl: f64,
    pub dt_max_s: f64,
    pub h_dry: f64,
    pub h_eps: f64,
    pub rain: Hyetograph,
    /// Infiltration capacity (m/s).
    pub infil_capacity: f64,
    pub initial_depth: f64,
    pub initial_stage: Option<f64>,
    pub kernel: KernelKind,
    pub skip_dry: bool,
    /// Baseline-locked Δt schedule (§7.11.2): the baseline's Δt per step. Each step is
    /// verified against `cfl_hard`; on violation the run restarts adaptively from the last
    /// sync snapshot.
    pub locked_dt: Option<Arc<Vec<f64>>>,
    /// Hard positivity limit for the unsplit scheme (0.5).
    pub cfl_hard: f64,
    /// Multiplier on inflow hydrographs (scenario ensembles, §6.6).
    pub inflow_scale: f64,
}

impl SolverParams {
    pub fn from_scenario(sc: &Scenario) -> Self {
        let h = &sc.hydrology;
        let cap = match h.infiltration {
            InfiltrationCfg::None => 0.0,
            InfiltrationCfg::ConstantCapacity { capacity_mm_h } => capacity_mm_h / 1000.0 / 3600.0,
        };
        Self {
            duration_s: h.duration_s,
            sync_interval_s: h.sync_interval_s,
            cfl: sc.solver.cfl,
            dt_max_s: sc.solver.dt_max_s,
            h_dry: sc.solver.h_dry_m,
            h_eps: sc.solver.h_eps_m,
            rain: Hyetograph::from_mm_h(&h.rainfall_hyetograph),
            infil_capacity: cap,
            initial_depth: h.initial_depth_m,
            initial_stage: h.initial_stage_m,
            kernel: KernelKind::Fused,
            skip_dry: true,
            locked_dt: None,
            cfl_hard: 0.5,
            inflow_scale: 1.0,
        }
    }
}

/// Snapshot of the full run state at a sync point (baseline warm-start, §7.6).
#[derive(Clone)]
pub struct Checkpoint<T: Real> {
    pub step: u64,
    pub t: f64,
    pub sync_index: u32,
    pub dt_next: f64,
    /// CFL rate of this state (for locked-Δt verification after a restore).
    pub smax: f64,
    pub h: AlignedBuf<T>,
    pub qx: AlignedBuf<T>,
    pub qy: AlignedBuf<T>,
    pub row_dry: Vec<bool>,
    pub mon_max: Vec<f64>,
    pub mon_tmax: Vec<f64>,
    pub acc: Accum,
}

/// Baseline recording: checkpoints + per padded row first step at which it held water.
#[derive(Default)]
pub struct BaselineRecord<T: Real> {
    pub checkpoints: Vec<Checkpoint<T>>,
    pub first_wet_step: Vec<u64>,
    /// Δt of every step (for baseline-locked candidates).
    pub dts: Vec<f64>,
    /// Optional subdomain-replay tape (§7.11.3), recorded when set before the run.
    pub tape: Option<crate::replay::Tape<T>>,
}

impl<T: Real> BaselineRecord<T> {
    /// Latest checkpoint valid for an edit touching padded rows `rows` (inclusive):
    /// all those rows must have been exactly dry in every state before the checkpoint.
    pub fn best_for(&self, rows: Option<(usize, usize)>) -> Option<&Checkpoint<T>> {
        let (a, b) = rows?;
        let w = (a..=b).map(|j| self.first_wet_step.get(j).copied().unwrap_or(0)).min()?;
        self.checkpoints.iter().filter(|c| c.step <= w).max_by_key(|c| c.step)
    }
    pub fn bytes(&self) -> usize {
        self.checkpoints.iter().map(|c| (c.h.len() * 3) * core::mem::size_of::<T>()).sum()
    }
}

/// Running accumulators (part of a checkpoint so warm-started runs continue exactly).
#[derive(Clone, Debug, Default)]
pub struct Accum {
    pub rows: Vec<LedgerRow>,
    pub v0: f64,
    pub rain: f64,
    pub inflow: f64,
    pub outflow: f64,
    pub infil: f64,
    pub fix: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub dt_sum: f64,
    pub cell_updates: u64,
    pub rows_skipped: u64,
    pub lts: f64,
}

pub struct Workspace<T: Real> {
    pub layout: Layout,
    pub h: AlignedBuf<T>,
    pub qx: AlignedBuf<T>,
    pub qy: AlignedBuf<T>,
    pub scratch: KernelScratch<T>,
    pub mon: Monitors,
    pub pool: Option<crate::pool::GridPool>,
    /// When set, the run records checkpoints (baseline only) within this byte budget.
    pub record: Option<(usize, BaselineRecord<T>)>,
    pub z_level: Vec<f64>,
    /// Subdomain replay: ghost ring from a baseline tape; band checked at sync points.
    pub replay: Option<(Arc<crate::replay::Tape<T>>, f64, f64)>,
    /// Restore point for baseline-locked Δt fallback (reused buffers).
    snap: Option<Checkpoint<T>>,
    /// Current grid-parallel thread count.
    pub threads: usize,
    /// Tail re-partitioning (§7.11.5): queried at every sync point; if it returns a
    /// different thread count the run is re-split across that many threads (results are
    /// bitwise unchanged, D0).
    pub threads_hint: Option<Arc<dyn Fn() -> usize + Send + Sync>>,
}

/// Worker set for `threads`, never larger than the number of strips.
fn grid_pool(threads: usize, ny: usize, strip_rows: usize) -> Option<crate::pool::GridPool> {
    let n = threads.min(ny.div_ceil(strip_rows.max(1)));
    (n > 1).then(|| crate::pool::GridPool::new(n))
}

impl<T: Real> Workspace<T> {
    pub fn new(layout: Layout, monitors: &MonitorSet, threads: usize, strip_rows: usize) -> Self {
        let pool = grid_pool(threads, layout.ny, strip_rows);
        let strip = if threads > 1 { strip_rows } else { layout.ny.max(1) };
        Self {
            layout,
            h: AlignedBuf::new(layout.len(), T::ZERO),
            qx: AlignedBuf::new(layout.len(), T::ZERO),
            qy: AlignedBuf::new(layout.len(), T::ZERO),
            scratch: KernelScratch::new(&layout, strip),
            mon: Monitors::new(&monitors.cells, layout.nx, layout.ny),
            pool,
            record: None,
            z_level: vec![],
            replay: None,
            snap: None,
            threads: threads.max(1),
            threads_hint: None,
        }
    }
    /// Switch thread count / strip height between runs (results are unchanged, D0).
    pub fn set_threads(&mut self, threads: usize, strip_rows: usize) {
        self.pool = None; // join the old workers before spawning new ones
        self.pool = grid_pool(threads, self.layout.ny, strip_rows);
        self.threads = threads.max(1);
        let strip = if threads > 1 { strip_rows } else { self.layout.ny.max(1) };
        self.scratch = KernelScratch::new(&self.layout, strip);
    }

    /// Re-split a run in progress (at a sync point, monitors gathered) across `threads`.
    fn repartition(&mut self, threads: usize, fused: bool) {
        let row_dry = std::mem::take(&mut self.scratch.row_dry);
        let ghost_wet = std::mem::take(&mut self.scratch.ghost_wet);
        self.set_threads(threads, 16);
        self.scratch.row_dry = row_dry;
        self.scratch.ghost_wet = ghost_wet;
        if fused {
            self.scratch.scatter_monitors(&self.mon);
        }
    }
    pub fn take_record(&mut self) -> Option<BaselineRecord<T>> {
        self.record.take().map(|r| r.1)
    }
}

struct WsView<'a, T: Real> {
    l: Layout,
    h: &'a [T],
    qx: &'a [T],
    qy: &'a [T],
}

impl<T: Real> StateAccess for WsView<'_, T> {
    fn nx(&self) -> usize {
        self.l.nx
    }
    fn ny(&self) -> usize {
        self.l.ny
    }
    fn depth(&self, out: &mut [f32]) {
        for r in 0..self.l.ny {
            for c in 0..self.l.nx {
                out[r * self.l.nx + c] = self.h[self.l.at(c + 1, r + 1)].to_f64() as f32;
            }
        }
    }
    fn discharge(&self, qx: &mut [f32], qy: &mut [f32]) {
        for r in 0..self.l.ny {
            for c in 0..self.l.nx {
                let k = self.l.at(c + 1, r + 1);
                qx[r * self.l.nx + c] = self.qx[k].to_f64() as f32;
                // Internal +row is south; report north-positive.
                qy[r * self.l.nx + c] = -(self.qy[k].to_f64() as f32);
            }
        }
    }
}

pub struct Solver<T: Real> {
    pub p: SolverParams,
    _t: core::marker::PhantomData<T>,
}

fn volume<T: Real>(l: &Layout, h: &[T], area: f64) -> f64 {
    let mut v = 0.0;
    for j in 1..=l.ny {
        let mut s = FixedSum::<T>::default();
        let r = l.row(j);
        let row = &h[r];
        for i in 1..=l.nx {
            s.add(i, row[i]);
        }
        v += s.total().to_f64();
    }
    v * area
}

impl<T: Real> Solver<T> {
    pub fn new(p: SolverParams) -> Self {
        Self { p, _t: core::marker::PhantomData }
    }

    fn consts(&self, terr: &PreparedTerrain<T>, dt: f64, rain: f64) -> StepConsts<T> {
        let b = &terr.base;
        StepConsts {
            g: T::from_f64(b.g),
            g_half: T::from_f64(0.5 * b.g),
            dt: T::from_f64(dt),
            dtdx: T::from_f64(dt / b.dx),
            dtdy: T::from_f64(dt / b.dy),
            inv_dx: T::from_f64(1.0 / b.dx),
            inv_dy: T::from_f64(1.0 / b.dy),
            h_dry: T::from_f64(self.p.h_dry),
            heps2: T::from_f64(self.p.h_eps * self.p.h_eps),
            q_min: T::from_f64(1e-30),
            rain: T::from_f64(rain),
            cap: T::from_f64(self.p.infil_capacity * dt),
        }
    }

    fn init_state(&self, ws: &mut Workspace<T>, terr: &PreparedTerrain<T>) {
        let l = ws.layout;
        ws.h.fill(T::ZERO);
        ws.qx.fill(T::ZERO);
        ws.qy.fill(T::ZERO);
        let b = &terr.base;
        let z = &ws.z_level;
        for r in 0..l.ny {
            for c in 0..l.nx {
                let k = r * l.nx + c;
                if b.wall.data[k] {
                    continue;
                }
                let mut d = self.p.initial_depth;
                if let Some(s) = self.p.initial_stage {
                    d = d.max(s - b.fine.z_ref - z[k]);
                }
                ws.h[l.at(c + 1, r + 1)] = T::from_f64(d.max(0.0)).canon_zero();
            }
        }
        for j in 0..l.ny + 2 {
            let rr = l.row(j);
            ws.scratch.row_dry[j] = ws.h[rr].iter().all(|v| *v == T::ZERO);
        }
        ws.mon.reset();
    }

    /// Initial CFL rate from the current state (one pre-pass, §5.2.1).
    fn initial_smax(&self, ws: &Workspace<T>, terr: &PreparedTerrain<T>) -> f64 {
        let l = ws.layout;
        let (g, dx, dy) = (terr.base.g, terr.base.dx, terr.base.dy);
        let mut s: f64 = 0.0;
        for j in 1..=l.ny {
            for i in 1..=l.nx {
                let k = l.at(i, j);
                let h = ws.h[k].to_f64();
                if h <= self.p.h_dry {
                    continue;
                }
                let c = (g * h).sqrt();
                let r = (ws.qx[k].to_f64().abs() / h + c) / dx + (ws.qy[k].to_f64().abs() / h + c) / dy;
                s = s.max(r);
            }
        }
        s
    }

    fn initial_smax_li(&self, ws: &Workspace<T>, terr: &PreparedTerrain<T>) -> f64 {
        let l = ws.layout;
        let b = &terr.base;
        let mut hmax: f64 = 0.0;
        for j in 1..=l.ny {
            for i in 1..=l.nx {
                hmax = hmax.max(ws.h[l.at(i, j)].to_f64());
            }
        }
        (b.g * hmax).sqrt() * 0.7 / b.dx.min(b.dy)
    }

    pub fn run_impl<O: Observer>(
        &self,
        ws: &mut Workspace<T>,
        terr: &PreparedTerrain<T>,
        start: Option<&Checkpoint<T>>,
        observer: &mut O,
    ) -> Result<RunSummary, SimulationError> {
        let t0 = Instant::now();
        let l = ws.layout;
        if l != terr.base.layout {
            return Err(SimulationError("workspace layout does not match terrain".into()));
        }
        ws.z_level = terr.z_level();
        let area = terr.base.cell_area();
        let p = &self.p;
        let mut acc;
        let (mut t, mut step, mut sync_index, mut dt_cfl, mut s_cur);
        let warm_steps;
        match start {
            Some(c) => {
                ws.h.copy_from_slice(&c.h);
                ws.qx.copy_from_slice(&c.qx);
                ws.qy.copy_from_slice(&c.qy);
                ws.scratch.row_dry.copy_from_slice(&c.row_dry);
                ws.mon.max.copy_from_slice(&c.mon_max);
                ws.mon.tmax.copy_from_slice(&c.mon_tmax);
                acc = c.acc.clone();
                t = c.t;
                step = c.step;
                sync_index = c.sync_index;
                dt_cfl = c.dt_next;
                s_cur = c.smax;
                warm_steps = c.step;
            }
            None => {
                self.init_state(ws, terr);
                acc = Accum { dt_min: f64::MAX, ..Default::default() };
                acc.v0 = volume(&l, &ws.h, area);
                t = 0.0;
                step = 0;
                sync_index = 0;
                let s0 = if p.kernel == KernelKind::LocalInertial { self.initial_smax_li(ws, terr) } else { self.initial_smax(ws, terr) };
                s_cur = s0;
                dt_cfl = if s0 > 0.0 { (p.cfl / s0).min(p.dt_max_s) } else { p.dt_max_s };
                warm_steps = 0;
            }
        }
        let mut first_wet: Vec<u64> = vec![];
        if let Some((_, rec)) = &mut ws.record {
            if rec.first_wet_step.len() != l.ny + 2 {
                rec.first_wet_step = vec![u64::MAX; l.ny + 2];
            }
            first_wet = std::mem::take(&mut rec.first_wet_step);
            for j in 0..l.ny + 2 {
                if !ws.scratch.row_dry[j] && first_wet[j] == u64::MAX {
                    first_wet[j] = step;
                }
            }
            // Rows adjacent to water-*creating* boundaries (stage, inflow) may receive water
            // from ghosts at any time; treat them as wet from the start. Walls and
            // transmissive ghosts only mirror/copy interior water, so they cannot wet a
            // dry row (conservative, keeps warm start exact).
            let b = &terr.base.boundaries;
            use crate::terrain::Ghost;
            let open = |g: &Ghost| matches!(g, Ghost::Stage(_) | Ghost::Inflow(..));
            for j in 1..=l.ny {
                if open(&b.west[j - 1]) || open(&b.east[j - 1]) {
                    first_wet[j] = first_wet[j].min(step);
                }
            }
            if b.north.iter().any(open) {
                first_wet[0] = first_wet[0].min(step);
                first_wet[1] = first_wet[1].min(step);
            }
            if b.south.iter().any(open) {
                first_wet[l.ny] = first_wet[l.ny].min(step);
                first_wet[l.ny + 1] = first_wet[l.ny + 1].min(step);
            }
            // Checkpoint at the initial state.
            if start.is_none() {
                let ck = self.checkpoint(ws, &acc, t, step, sync_index, dt_cfl, s_cur);
                if let Some((_, rec)) = &mut ws.record {
                    rec.checkpoints.push(ck);
                }
            }
        }
        if p.kernel == KernelKind::Fused {
            ws.scratch.scatter_monitors(&ws.mon);
        }
        let n_sync = (p.duration_s / p.sync_interval_s).ceil() as u32;
        let cells = (l.nx * l.ny) as u64;
        let mut aborted = None;
        let z_level = ws.z_level.clone();
        let zfun = |c: usize, r: usize| z_level[r * l.nx + c];
        let mut locked = p.locked_dt.clone();
        let mut li_faces: Vec<T> = Vec::with_capacity(l.nx + l.ny);
        let mut li_old: Vec<f64> = Vec::new();
        let mut dt_fallback_t = None;
        let replay = ws.replay.clone();
        if locked.is_some() {
            let ck = self.checkpoint(ws, &acc, t, step, sync_index, dt_cfl, s_cur);
            ws.snap = Some(ck);
        }
        while t < p.duration_s && sync_index < n_sync {
            let t_sync = ((sync_index + 1) as f64 * p.sync_interval_s).min(p.duration_s);
            let mut dt = dt_cfl.min(p.dt_max_s);
            if let Some(sched) = &locked {
                // Baseline-locked Δt (§7.11.2): verified against the hard CFL limit.
                match sched.get(step as usize) {
                    Some(&d) if d * s_cur <= p.cfl_hard => dt = d,
                    _ if replay.is_some() => {
                        aborted = Some(AbortReason::ReplayInvalid { sync_index, reason: "locked-Δt verification failed".into() });
                        break;
                    }
                    _ => {
                        // Restart adaptively from the last sync snapshot.
                        let c = ws.snap.take().expect("locked snapshot");
                        ws.h.copy_from_slice(&c.h);
                        ws.qx.copy_from_slice(&c.qx);
                        ws.qy.copy_from_slice(&c.qy);
                        ws.scratch.row_dry.copy_from_slice(&c.row_dry);
                        ws.mon.max.copy_from_slice(&c.mon_max);
                        ws.mon.tmax.copy_from_slice(&c.mon_tmax);
                        if p.kernel == KernelKind::Fused {
                            ws.scratch.scatter_monitors(&ws.mon);
                        }
                        acc = c.acc.clone();
                        t = c.t;
                        step = c.step;
                        sync_index = c.sync_index;
                        dt_cfl = c.dt_next;
                        s_cur = c.smax;
                        dt_fallback_t = Some(t);
                        locked = None;
                        continue;
                    }
                }
            }
            let hits_sync = t + dt >= t_sync - 1e-9 * p.sync_interval_s;
            if hits_sync {
                dt = t_sync - t;
            }
            let rain = p.rain.integral(t, t + dt);
            let t_new = if hits_sync { t_sync } else { t + dt };
            match &replay {
                Some((tape, _, _)) => {
                    if !tape.fill(step as usize, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.scratch.ghost_wet) {
                        aborted = Some(AbortReason::ReplayInvalid { sync_index, reason: "tape exhausted".into() });
                        break;
                    }
                }
                None if p.kernel == KernelKind::LocalInertial => {
                    crate::inertial::save_boundary_faces(&l, &ws.qx, &ws.qy, &mut li_faces);
                    fill_ghosts(&l, &terr.base, &zfun, t, p.inflow_scale, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.scratch.ghost_wet);
                    crate::inertial::restore_boundary_faces(&l, &terr.base.boundaries, &mut ws.qx, &mut ws.qy, &li_faces);
                }
                None => fill_ghosts(&l, &terr.base, &zfun, t, p.inflow_scale, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.scratch.ghost_wet),
            }
            if let Some((_, rec)) = &mut ws.record {
                if let Some(tp) = &mut rec.tape {
                    tp.record_ring(&ws.h, &ws.qx, &ws.qy);
                }
                rec.dts.push(dt);
            }
            let cx = StepCtx {
                l: &l,
                terr,
                k: self.consts(terr, dt, rain),
                t_new,
                dt,
                want_sum_s: hits_sync,
                skip_dry: p.skip_dry,
            };
            let st = match p.kernel {
                KernelKind::Oracle => step_oracle(&cx, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.scratch, &mut ws.mon),
                KernelKind::Fused => step_fused(&cx, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.scratch, &mut ws.mon, ws.pool.as_ref()),
                KernelKind::LocalInertial => {
                    let mut s = crate::inertial::step_local_inertial(&cx, &mut ws.h, &mut ws.qx, &mut ws.qy, &mut ws.mon, &mut ws.scratch.row_dry, &mut li_old);
                    // Depth-valued sums → the caller's convention (per unit area).
                    s.sum_s = if hits_sync { s.sum_s } else { 0.0 };
                    s
                }
            };
            step += 1;
            t = t_new;
            acc.rain += rain * area * terr.base.wet_count as f64;
            acc.inflow += st.bnd_in;
            acc.outflow += st.bnd_out;
            acc.infil += st.infil * area;
            acc.fix += st.fix * area;
            acc.dt_min = acc.dt_min.min(dt);
            acc.dt_max = acc.dt_max.max(dt);
            acc.dt_sum += dt;
            acc.cell_updates += cells - st.rows_skipped * l.nx as u64;
            acc.rows_skipped += st.rows_skipped;
            if st.bad {
                aborted = Some(AbortReason::Unhealthy { message: format!("negative depth or NaN at t={t:.3}s, step {step}") });
                break;
            }
            dt_cfl = if st.smax > 0.0 { p.cfl / st.smax } else { p.dt_max_s };
            s_cur = st.smax;
            if !first_wet.is_empty() {
                for j in 0..l.ny + 2 {
                    if first_wet[j] == u64::MAX && !(ws.scratch.row_dry[j] && !ws.scratch.ghost_wet[j]) {
                        first_wet[j] = step;
                    }
                }
            }
            if hits_sync {
                if p.kernel == KernelKind::Fused {
                    ws.scratch.gather_monitors(&mut ws.mon);
                }
                sync_index += 1;
                if st.sum_s > 0.0 {
                    acc.lts = acc.lts.max(st.smax * cells as f64 / st.sum_s);
                }
                let v = volume(&l, &ws.h, area);
                let expected = acc.v0 + acc.rain + acc.inflow - acc.outflow - acc.infil + acc.fix;
                acc.rows.push(LedgerRow {
                    t,
                    volume_m3: v,
                    rain_m3: acc.rain,
                    inflow_m3: acc.inflow,
                    outflow_m3: acc.outflow,
                    infiltration_m3: acc.infil,
                    positivity_fix_m3: acc.fix,
                    residual_m3: v - expected,
                });
                if let Some((_, rec)) = &mut ws.record
                    && let Some(tp) = &mut rec.tape {
                        tp.record_band(&ws.h, &ws.qx, &ws.qy);
                    }
                if let Some((tape, tol_h, tol_q)) = &replay {
                    match tape.band_diff(sync_index, &ws.h, &ws.qx, &ws.qy) {
                        Some((dh, dq)) if dh <= *tol_h && dq <= *tol_q => {}
                        r => {
                            aborted = Some(AbortReason::ReplayInvalid {
                                sync_index,
                                reason: match r {
                                    Some((dh, dq)) => format!("edit influence reached the ROI band (|Δh| {dh:.2e} m, |Δq| {dq:.2e} m²/s)"),
                                    None => "tape has no band for this sync".into(),
                                },
                            });
                            break;
                        }
                    }
                }
                let view = SyncView {
                    t,
                    step,
                    sync_index,
                    monitor_max: &ws.mon.max,
                    state: &WsView { l, h: &ws.h, qx: &ws.qx, qy: &ws.qy },
                };
                if let ControlFlow::Break(r) = observer.on_sync(&view) {
                    aborted = Some(r);
                    break;
                }
                let want_ck = match &ws.record {
                    Some((budget, rec)) => rec.bytes() + 3 * l.len() * core::mem::size_of::<T>() <= *budget,
                    None => false,
                };
                if let Some(hint) = ws.threads_hint.clone() {
                    // Grow whenever cores free up; shrink only on a 2× change (avoids churn).
                    let want = hint().max(1);
                    if want > ws.threads || want * 2 <= ws.threads {
                        ws.repartition(want, p.kernel == KernelKind::Fused);
                    }
                }
                if locked.is_some() {
                    let ck = ws.snap.take();
                    ws.snap = Some(self.checkpoint_into(ck, ws, &acc, t, step, sync_index, dt_cfl, s_cur));
                }
                if want_ck {
                    let ck = self.checkpoint(ws, &acc, t, step, sync_index, dt_cfl, s_cur);
                    if let Some((_, rec)) = &mut ws.record {
                        rec.checkpoints.push(ck);
                    }
                }
            }
        }
        if p.kernel == KernelKind::Fused {
            ws.scratch.gather_monitors(&mut ws.mon);
        }
        if let Some((_, rec)) = &mut ws.record {
            rec.first_wet_step = first_wet;
        }
        let last = acc.rows.last().cloned().unwrap_or_default();
        let gross = acc.v0 + acc.rain + acc.inflow;
        let ledger = LedgerReport {
            v0_m3: acc.v0,
            rel_residual: last.residual_m3.abs() / gross.max(1e-300),
            abs_residual_m3: last.residual_m3.abs(),
            gross_throughput_m3: gross,
            rows: acc.rows.clone(),
        };
        Ok(RunSummary {
            t_end: t,
            steps: step,
            warm_start_steps: warm_steps,
            sync_count: sync_index,
            aborted,
            monitor_max: ws.mon.max.clone(),
            monitor_tmax: ws.mon.tmax.clone(),
            ledger,
            dt_min: if acc.dt_min == f64::MAX { 0.0 } else { acc.dt_min },
            dt_max: acc.dt_max,
            dt_mean: if step > 0 { acc.dt_sum / step as f64 } else { 0.0 },
            cell_updates: acc.cell_updates,
            rows_skipped: acc.rows_skipped,
            wall_s: t0.elapsed().as_secs_f64(),
            lts_potential: acc.lts,
            dt_fallback_t,
        })
    }

    /// Like `checkpoint`, reusing the buffers of `prev` (no allocation after the first).
    #[allow(clippy::too_many_arguments)]
    fn checkpoint_into(&self, prev: Option<Checkpoint<T>>, ws: &Workspace<T>, acc: &Accum, t: f64, step: u64, sync_index: u32, dt_next: f64, smax: f64) -> Checkpoint<T> {
        match prev {
            Some(mut c) => {
                c.step = step;
                c.t = t;
                c.sync_index = sync_index;
                c.dt_next = dt_next;
                c.smax = smax;
                c.h.copy_from_slice(&ws.h);
                c.qx.copy_from_slice(&ws.qx);
                c.qy.copy_from_slice(&ws.qy);
                c.row_dry.clone_from(&ws.scratch.row_dry);
                c.mon_max.clone_from(&ws.mon.max);
                c.mon_tmax.clone_from(&ws.mon.tmax);
                c.acc.clone_from(acc);
                c
            }
            None => self.checkpoint(ws, acc, t, step, sync_index, dt_next, smax),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn checkpoint(&self, ws: &Workspace<T>, acc: &Accum, t: f64, step: u64, sync_index: u32, dt_next: f64, smax: f64) -> Checkpoint<T> {
        Checkpoint {
            step,
            t,
            sync_index,
            dt_next,
            smax,
            h: ws.h.clone(),
            qx: ws.qx.clone(),
            qy: ws.qy.clone(),
            row_dry: ws.scratch.row_dry.clone(),
            mon_max: ws.mon.max.clone(),
            mon_tmax: ws.mon.tmax.clone(),
            acc: acc.clone(),
        }
    }
}

impl<T: Real> ForwardModel for Solver<T> {
    type Terrain = PreparedTerrain<T>;
    type Ws = Workspace<T>;
    type Checkpoint = Checkpoint<T>;

    fn workspace(&self, terrain: &PreparedTerrain<T>, monitors: &MonitorSet) -> Workspace<T> {
        Workspace::new(terrain.base.layout, monitors, 1, 32)
    }

    fn run<O: Observer>(
        &self,
        ws: &mut Workspace<T>,
        terrain: &PreparedTerrain<T>,
        start: Option<&Checkpoint<T>>,
        observer: &mut O,
    ) -> Result<RunSummary, SimulationError> {
        self.run_impl(ws, terrain, start, observer)
    }
}
