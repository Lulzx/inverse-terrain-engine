//! Solver-facing contracts (spec §3.2). Streaming, caller-allocated, no trajectories.

use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// Interior cells (linear index `row * nx + col`) tracked in the hot loop. Sorted, unique.
#[derive(Clone, Debug, Default)]
pub struct MonitorSet {
    pub cells: Vec<u32>,
}

impl MonitorSet {
    pub fn new(mut cells: Vec<u32>) -> Self {
        cells.sort_unstable();
        cells.dedup();
        Self { cells }
    }
    pub fn all(nx: usize, ny: usize) -> Self {
        Self { cells: (0..(nx * ny) as u32).collect() }
    }
    pub fn position(&self, cell: u32) -> Option<usize> {
        self.cells.binary_search(&cell).ok()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AbortReason {
    /// Guard cell running max exceeded baseline + tolerance at this sync point (exact, §7.6).
    GuardViolation { cell: u32, exceedance_m: f64, sync_index: u32 },
    /// Running objective already >= frozen incumbent (elitist methods only).
    Incumbent { lower_bound: f64 },
    /// Subdomain replay could not be trusted (edit influence reached the ROI band, or the
    /// locked Δt could not be kept); the caller escalates to a full-domain run (§7.11.3).
    ReplayInvalid { sync_index: u32, reason: String },
    /// Health check failed (negative depth / NaN). No rollback (§5.3).
    Unhealthy { message: String },
}

/// Read access to the current state at a sync point.
pub trait StateAccess {
    fn nx(&self) -> usize;
    fn ny(&self) -> usize;
    /// Interior depth, row-major north-up, as f32.
    fn depth(&self, out: &mut [f32]);
    /// Interior unit discharges (east, north-positive) as f32.
    fn discharge(&self, qx: &mut [f32], qy: &mut [f32]);
}

pub struct SyncView<'a> {
    pub t: f64,
    pub step: u64,
    pub sync_index: u32,
    /// Running max depth at monitored cells (same order as `MonitorSet::cells`).
    pub monitor_max: &'a [f64],
    pub state: &'a dyn StateAccess,
}

pub trait Observer {
    fn on_sync(&mut self, view: &SyncView<'_>) -> ControlFlow<AbortReason>;
}

pub struct NoopObserver;
impl Observer for NoopObserver {
    fn on_sync(&mut self, _: &SyncView<'_>) -> ControlFlow<AbortReason> {
        ControlFlow::Continue(())
    }
}

/// One row of the mass ledger, per sync interval (spec §5.4).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LedgerRow {
    pub t: f64,
    pub volume_m3: f64,
    pub rain_m3: f64,
    pub inflow_m3: f64,
    pub outflow_m3: f64,
    pub infiltration_m3: f64,
    /// Volume added by clamping roundoff-negative depths to zero (explicit, never hidden).
    pub positivity_fix_m3: f64,
    pub residual_m3: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LedgerReport {
    pub v0_m3: f64,
    pub rows: Vec<LedgerRow>,
    /// |residual| / max(gross throughput, tiny)
    pub rel_residual: f64,
    pub abs_residual_m3: f64,
    pub gross_throughput_m3: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunSummary {
    pub t_end: f64,
    pub steps: u64,
    /// Steps skipped by warm start (state restored from a baseline checkpoint).
    pub warm_start_steps: u64,
    pub sync_count: u32,
    pub aborted: Option<AbortReason>,
    pub monitor_max: Vec<f64>,
    pub monitor_tmax: Vec<f64>,
    pub ledger: LedgerReport,
    pub dt_min: f64,
    pub dt_max: f64,
    pub dt_mean: f64,
    pub cell_updates: u64,
    pub rows_skipped: u64,
    pub wall_s: f64,
    /// Ideal local-time-stepping work ratio N·max(s)/Σs, max over sync points (§7.11.2).
    pub lts_potential: f64,
    /// Baseline-locked Δt only: simulated time of the sync checkpoint from which the run
    /// fell back to adaptive Δt after a hard-CFL violation (§7.11.2). `None` = no fallback.
    #[serde(default)]
    pub dt_fallback_t: Option<f64>,
}

#[derive(Debug)]
pub struct SimulationError(pub String);
impl core::fmt::Display for SimulationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SimulationError {}

/// Forward model contract. The optimizer depends only on this trait.
pub trait ForwardModel: Sync {
    type Terrain: Sync;
    type Ws: Send;
    type Checkpoint: Send + Sync;

    fn workspace(&self, terrain: &Self::Terrain, monitors: &MonitorSet) -> Self::Ws;

    fn run<O: Observer>(
        &self,
        ws: &mut Self::Ws,
        terrain: &Self::Terrain,
        start: Option<&Self::Checkpoint>,
        observer: &mut O,
    ) -> Result<RunSummary, SimulationError>;
}

pub fn control_continue() -> ControlFlow<AbortReason> {
    ControlFlow::Continue(())
}
