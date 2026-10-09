//! Damage proxies, earthwork cost and guard constraints (spec §6.3), independent of
//! the optimizer. Transcendentals come from the pinned pure-Rust `libm` (§7.9).

use crate::design::TerrainEdit;
use crate::model::{MonitorSet, SyncView};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct Asset {
    pub name: String,
    /// Monitored cells representing the asset (footprint, or outer ring in wall mode).
    pub cells: Vec<u32>,
    pub threshold_m: f64,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct ObjectiveDef {
    pub assets: Vec<Asset>,
    pub guard_cells: Vec<u32>,
    pub guard_tolerance_m: f64,
    pub tau_m: f64,
    pub cut_cost: f64,
    pub fill_cost: f64,
    pub earth_weight: f64,
    pub require_no_offsite_worsening: bool,
    pub monitor_full_domain: bool,
    pub nx: usize,
    pub ny: usize,
}

/// Positions of each asset / guard cell inside the monitor list.
#[derive(Clone, Debug)]
pub struct Objective {
    pub def: ObjectiveDef,
    pub monitors: MonitorSet,
    asset_pos: Vec<Vec<usize>>,
    guard_pos: Vec<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssetReport {
    pub name: String,
    pub peak_depth_m: f64,
    pub threshold_m: f64,
    pub exceedance_m: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObjectiveReport {
    /// Scalar search objective J = J_risk + λ·J_earth (disclosed normalization).
    pub j: f64,
    pub j_risk: f64,
    pub j_earth: f64,
    pub cut_m3: f64,
    pub fill_m3: f64,
    pub assets: Vec<AssetReport>,
    /// Sum of true exceedances (reporting metric, not the surrogate).
    pub total_exceedance_m: f64,
    pub assets_exceeding: usize,
    /// max over guard cells of D_θ − D_0 (m); 0 if no guards.
    pub guard_max_worsening_m: f64,
    /// Count of guard cells worse than tolerance.
    pub guard_cells_worsened: usize,
    /// > 0 when infeasible (m of exceedance beyond tolerance).
    pub guard_violation_m: f64,
    pub feasible: bool,
}

#[inline]
fn softplus(x: f64) -> f64 {
    // Stable log(1 + e^x).
    if x > 30.0 { x } else if x < -30.0 { libm::exp(x) } else { libm::log1p(libm::exp(x)) }
}

impl Objective {
    pub fn new(def: ObjectiveDef) -> Self {
        let mut cells: Vec<u32> = def.assets.iter().flat_map(|a| a.cells.iter().copied()).collect();
        cells.extend(def.guard_cells.iter().copied());
        let monitors = if def.monitor_full_domain { MonitorSet::all(def.nx, def.ny) } else { MonitorSet::new(cells) };
        let asset_pos = def
            .assets
            .iter()
            .map(|a| a.cells.iter().map(|c| monitors.position(*c).unwrap()).collect())
            .collect();
        let guard_pos = def.guard_cells.iter().map(|c| monitors.position(*c).unwrap()).collect();
        Self { def, monitors, asset_pos, guard_pos }
    }

    pub fn asset_peaks(&self, monitor_max: &[f64]) -> Vec<f64> {
        self.asset_pos.iter().map(|p| p.iter().map(|&k| monitor_max[k]).fold(0.0, f64::max)).collect()
    }

    /// Evaluate a finished run against the baseline's monitor maxima (same MonitorSet).
    pub fn evaluate(&self, monitor_max: &[f64], edit: &TerrainEdit, baseline_max: Option<&[f64]>) -> ObjectiveReport {
        let d = &self.def;
        let peaks = self.asset_peaks(monitor_max);
        let mut rep = ObjectiveReport::default();
        for (a, &pk) in d.assets.iter().zip(&peaks) {
            let ex = (pk - a.threshold_m).max(0.0);
            rep.j_risk += a.weight * d.tau_m * softplus((pk - a.threshold_m) / d.tau_m);
            rep.total_exceedance_m += ex;
            if ex > 0.0 {
                rep.assets_exceeding += 1;
            }
            rep.assets.push(AssetReport { name: a.name.clone(), peak_depth_m: pk, threshold_m: a.threshold_m, exceedance_m: ex });
        }
        rep.cut_m3 = edit.cut_m3;
        rep.fill_m3 = edit.fill_m3;
        rep.j_earth = d.cut_cost * edit.cut_m3 + d.fill_cost * edit.fill_m3;
        rep.j = rep.j_risk + d.earth_weight * rep.j_earth;
        if let Some(b) = baseline_max {
            let mut worst = 0.0f64;
            for &k in &self.guard_pos {
                let w = monitor_max[k] - b[k];
                worst = worst.max(w);
                if w > d.guard_tolerance_m {
                    rep.guard_cells_worsened += 1;
                }
            }
            rep.guard_max_worsening_m = worst;
            if d.require_no_offsite_worsening && worst > d.guard_tolerance_m {
                rep.guard_violation_m = worst - d.guard_tolerance_m;
            }
        }
        rep.feasible = rep.guard_violation_m == 0.0;
        rep
    }

    /// Exact early-abort test (spec §7.6): running maxima are monotone, so a guard cell
    /// already above baseline final max + tolerance stays infeasible.
    pub fn guard_breach(&self, view: &SyncView<'_>, baseline_max: &[f64]) -> Option<(u32, f64)> {
        if !self.def.require_no_offsite_worsening {
            return None;
        }
        let mut worst: Option<(u32, f64)> = None;
        for &k in &self.guard_pos {
            let w = view.monitor_max[k] - baseline_max[k] - self.def.guard_tolerance_m;
            if w > 0.0 && worst.is_none_or(|(_, v)| w > v) {
                worst = Some((self.monitors.cells[k], w));
            }
        }
        worst
    }

    /// Lower bound of J from running maxima (monotone): used for incumbent abort.
    pub fn running_lower_bound(&self, monitor_max: &[f64], edit: &TerrainEdit) -> f64 {
        self.evaluate(monitor_max, edit, None).j
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_monotone_and_reports_true_exceedance() {
        let def = ObjectiveDef {
            assets: vec![Asset { name: "a".into(), cells: vec![3, 4], threshold_m: 0.1, weight: 1.0 }],
            guard_cells: vec![7],
            guard_tolerance_m: 0.02,
            tau_m: 0.02,
            cut_cost: 1.0,
            fill_cost: 1.0,
            earth_weight: 0.0,
            require_no_offsite_worsening: true,
            monitor_full_domain: false,
            nx: 10,
            ny: 1,
        };
        let o = Objective::new(def);
        let e = TerrainEdit::default();
        let lo = o.evaluate(&[0.05, 0.12, 0.0], &e, Some(&[0.0, 0.0, 0.0]));
        let hi = o.evaluate(&[0.05, 0.20, 0.03], &e, Some(&[0.0, 0.0, 0.0]));
        assert!(hi.j > lo.j);
        assert!((lo.total_exceedance_m - 0.02).abs() < 1e-12);
        assert!(lo.feasible && !hi.feasible);
    }
}
