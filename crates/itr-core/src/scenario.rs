//! Scenario schema (spec §4.4). TOML with `deny_unknown_fields`; parsing is from a
//! string so this crate never touches the filesystem.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub schema_version: String,
    pub scenario_id: String,
    pub terrain: TerrainCfg,
    pub hydrology: HydrologyCfg,
    #[serde(default)]
    pub boundaries: BoundariesCfg,
    #[serde(default)]
    pub design: Option<DesignCfg>,
    #[serde(default)]
    pub objectives: Option<ObjectivesCfg>,
    #[serde(default)]
    pub solver: SolverCfg,
    #[serde(default)]
    pub optimizer: OptimizerCfg,
    /// Multi-scenario robust objective (§6.6). Absent = single nominal scenario.
    #[serde(default)]
    pub robustness: Option<RobustnessCfg>,
}

/// J_robust(θ) = E_s[J(θ, s)] + β·CVaR_α(J(θ, s)) over a declared, equally weighted
/// ensemble. Feasibility requires every member to be feasible (each against its own
/// baseline). CVaR is meaningful only when the member distribution is justified.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustnessCfg {
    #[serde(default = "r_beta")]
    pub beta: f64,
    #[serde(default = "r_alpha")]
    pub alpha: f64,
    /// Member 0 is the scenario as written scaled by its factors (normally 1).
    pub members: Vec<EnsembleMember>,
    /// Free-text justification of the ensemble (recorded in the manifest).
    #[serde(default)]
    pub uncertainty_model: String,
}
fn r_beta() -> f64 {
    0.5
}
fn r_alpha() -> f64 {
    0.8
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnsembleMember {
    pub name: String,
    #[serde(default = "one")]
    pub rain_scale: f64,
    #[serde(default = "one")]
    pub inflow_scale: f64,
}
fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerrainCfg {
    pub path: String,
    pub crs: String,
    pub vertical_datum: String,
    #[serde(default = "default_m")]
    pub elevation_units: String,
    #[serde(default)]
    pub expected_cell_size_m: Option<f64>,
    /// Optional Manning-n raster (same grid). Overrides `hydrology.manning_n`.
    #[serde(default)]
    pub manning_path: Option<String>,
    /// Optional solid obstacles (walls) as GeoJSON polygons.
    #[serde(default)]
    pub obstacles: Option<String>,
}

fn default_m() -> String {
    "m".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HydrologyCfg {
    pub duration_s: f64,
    pub sync_interval_s: f64,
    /// `[time_s, rate_mm_h]`, piecewise-linear.
    #[serde(default)]
    pub rainfall_hyetograph: Vec<[f64; 2]>,
    pub manning_n: f64,
    #[serde(default)]
    pub initial_depth_m: f64,
    /// Optional initial water-surface elevation (lake), applied where above ground.
    #[serde(default)]
    pub initial_stage_m: Option<f64>,
    pub infiltration: InfiltrationCfg,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "model", rename_all = "snake_case")]
pub enum InfiltrationCfg {
    None,
    ConstantCapacity { capacity_mm_h: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    North,
    South,
    East,
    West,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
/// `range_m = [a, b]` selects part of a side, in metres along the side measured from its
/// first cell: the north edge for west/east sides, the west edge for north/south sides.
pub enum BoundarySegment {
    /// Specified water-surface elevation in the ghost cells.
    Stage {
        side: Side,
        stage_m: f64,
        #[serde(default)]
        range_m: Option<[f64; 2]>,
    },
    /// Prescribed inflow hydrograph `[time_s, Q_m3_s]` spread uniformly over the segment.
    Inflow {
        side: Side,
        hydrograph: Vec<[f64; 2]>,
        #[serde(default)]
        range_m: Option<[f64; 2]>,
    },
    /// Zero-gradient (transmissive) boundary. Partially reflective for non-normal or
    /// subcritical inflow; labelled as such in reports (spec §4.2).
    Transmissive {
        side: Side,
        #[serde(default)]
        range_m: Option<[f64; 2]>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundariesCfg {
    #[serde(default = "default_wall")]
    pub default: String,
    #[serde(default)]
    pub segments: Vec<BoundarySegment>,
}

fn default_wall() -> String {
    "wall".into()
}

impl Default for BoundariesCfg {
    fn default() -> Self {
        Self { default: default_wall(), segments: vec![] }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizeCfg {
    #[serde(default = "q_h")]
    pub height_m: f64,
    #[serde(default = "q_p")]
    pub position_m: f64,
    #[serde(default = "q_p")]
    pub width_m: f64,
    #[serde(default = "q_a")]
    pub angle_deg: f64,
}
fn q_h() -> f64 {
    0.01
}
fn q_p() -> f64 {
    0.5
}
fn q_a() -> f64 {
    1.0
}
impl Default for QuantizeCfg {
    fn default() -> Self {
        Self { height_m: q_h(), position_m: q_p(), width_m: q_p(), angle_deg: q_a() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignCfg {
    pub editable_mask: String,
    pub primitives: String,
    pub max_abs_elevation_change_m: f64,
    pub max_earthwork_volume_m3: f64,
    #[serde(default)]
    pub quantize: QuantizeCfg,
    /// Max slope of the edit surface |∇Δz| (rise/run); 0 disables.
    #[serde(default = "d_slope")]
    pub max_edit_slope: f64,
    #[serde(default = "d_true")]
    pub require_no_offsite_worsening: bool,
    #[serde(default = "d_free")]
    pub placement: String,
    #[serde(default = "d_frac")]
    pub free_placement_fraction: f64,
    #[serde(default = "d_cost")]
    pub cut_cost_per_m3: f64,
    #[serde(default = "d_cost")]
    pub fill_cost_per_m3: f64,
}
fn d_slope() -> f64 {
    0.5
}
fn d_true() -> bool {
    true
}
fn d_free() -> String {
    "free".into()
}
fn d_frac() -> f64 {
    0.2
}
fn d_cost() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectivesCfg {
    pub protected_areas: String,
    #[serde(default)]
    pub downstream_guard_areas: Option<String>,
    pub depth_threshold_m: f64,
    #[serde(default = "d_guard")]
    pub guard_tolerance_m: f64,
    #[serde(default = "d_ground")]
    pub building_mode: String,
    #[serde(default)]
    pub monitor_full_domain: bool,
    /// Softplus smoothing scale τ (m).
    #[serde(default = "d_tau")]
    pub smoothing_m: f64,
    /// λ in J = J_risk + λ·J_earth. Units: (m of exceedance) per cost unit; disclosed in reports.
    #[serde(default = "d_lambda")]
    pub earthwork_weight: f64,
    /// When true, any guard cell is also "map-wide": all non-protected cells are guards.
    #[serde(default)]
    pub guard_whole_map: bool,
}
fn d_guard() -> f64 {
    0.02
}
fn d_ground() -> String {
    "ground".into()
}
fn d_tau() -> f64 {
    0.02
}
fn d_lambda() -> f64 {
    1e-5
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DtMode {
    Adaptive,
    BaselineLocked,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverCfg {
    #[serde(default = "s_method")]
    pub method: String,
    #[serde(default = "s_prec")]
    pub precision: Precision,
    #[serde(default = "s_backend")]
    pub backend: Backend,
    #[serde(default = "s_dtmode")]
    pub dt_mode: DtMode,
    #[serde(default = "s_cfl")]
    pub cfl: f64,
    #[serde(default = "s_dtmax")]
    pub dt_max_s: f64,
    #[serde(default = "s_hdry")]
    pub h_dry_m: f64,
    #[serde(default = "s_heps")]
    pub h_eps_m: f64,
    /// Subdomain replay (§7.11.3): "off" | "auto". Requires `dt_mode = "baseline_locked"`.
    #[serde(default = "s_replay")]
    pub subdomain_replay: String,
    /// Physics for coarse fidelity levels (§7.11.2): "hll" | "local_inertial". The final
    /// level (1) always uses HLL.
    #[serde(default = "s_screen")]
    pub screening_physics: String,
}
fn s_screen() -> String {
    "hll".into()
}
fn s_replay() -> String {
    "off".into()
}
fn s_method() -> String {
    "fv1_hll_hr".into()
}
fn s_prec() -> Precision {
    Precision::F32
}
fn s_backend() -> Backend {
    Backend::Cpu
}
fn s_dtmode() -> DtMode {
    DtMode::Adaptive
}
fn s_cfl() -> f64 {
    0.35
}
fn s_dtmax() -> f64 {
    5.0
}
fn s_hdry() -> f64 {
    1e-6
}
fn s_heps() -> f64 {
    1e-5
}
impl Default for SolverCfg {
    fn default() -> Self {
        Self {
            method: s_method(),
            precision: s_prec(),
            backend: s_backend(),
            dt_mode: s_dtmode(),
            cfl: s_cfl(),
            dt_max_s: s_dtmax(),
            h_dry_m: s_hdry(),
            h_eps_m: s_heps(),
            subdomain_replay: s_replay(),
            screening_physics: s_screen(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerCfg {
    #[serde(default = "o_method")]
    pub method: String,
    #[serde(default = "o_seed")]
    pub seed: u64,
    #[serde(default)]
    pub population: Option<usize>,
    #[serde(default = "o_sims")]
    pub max_simulations: usize,
    #[serde(default)]
    pub max_cell_updates: Option<u64>,
    #[serde(default = "o_levels")]
    pub fidelity_levels: Vec<usize>,
    /// Share of the simulation budget spent at each fidelity level (same length).
    #[serde(default)]
    pub level_budget_share: Option<Vec<f64>>,
    #[serde(default = "o_sigma")]
    pub initial_sigma: f64,
    #[serde(default = "o_attempts")]
    pub max_static_attempts: usize,
}
fn o_method() -> String {
    "cma_es".into()
}
fn o_seed() -> u64 {
    12345
}
fn o_sims() -> usize {
    200
}
fn o_levels() -> Vec<usize> {
    vec![1]
}
fn o_sigma() -> f64 {
    0.3
}
fn o_attempts() -> usize {
    50
}
impl Default for OptimizerCfg {
    fn default() -> Self {
        Self {
            method: o_method(),
            seed: o_seed(),
            population: None,
            max_simulations: o_sims(),
            max_cell_updates: None,
            fidelity_levels: o_levels(),
            level_budget_share: None,
            initial_sigma: o_sigma(),
            max_static_attempts: o_attempts(),
        }
    }
}

#[derive(Debug)]
pub struct ScenarioError(pub String);
impl core::fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ScenarioError {}

impl Scenario {
    pub fn from_toml(s: &str) -> Result<Self, ScenarioError> {
        let sc: Scenario = toml::from_str(s).map_err(|e| ScenarioError(e.to_string()))?;
        sc.validate()?;
        Ok(sc)
    }

    pub fn validate(&self) -> Result<(), ScenarioError> {
        let err = |m: &str| Err(ScenarioError(m.to_string()));
        if self.schema_version != "0.2" && self.schema_version != "0.1" {
            return err("unsupported schema_version (expected \"0.2\")");
        }
        if self.terrain.elevation_units != "m" {
            return err("terrain.elevation_units must be \"m\" (convert before import)");
        }
        let h = &self.hydrology;
        if !(h.duration_s > 0.0) || !(h.sync_interval_s > 0.0) {
            return err("hydrology.duration_s and sync_interval_s must be > 0");
        }
        if !(h.manning_n >= 0.0 && h.manning_n < 0.5) {
            return err("hydrology.manning_n out of range [0, 0.5)");
        }
        let mut last = f64::NEG_INFINITY;
        for p in &h.rainfall_hyetograph {
            if p[0] < last || p[1] < 0.0 {
                return err("rainfall_hyetograph must have increasing times and non-negative rates");
            }
            last = p[0];
        }
        if let InfiltrationCfg::ConstantCapacity { capacity_mm_h } = h.infiltration
            && capacity_mm_h < 0.0 {
                return err("infiltration capacity must be >= 0");
            }
        if self.boundaries.default != "wall" {
            return err("boundaries.default must be \"wall\" in v0.1; use segments for other kinds");
        }
        let s = &self.solver;
        if s.method != "fv1_hll_hr" {
            return err("solver.method: only \"fv1_hll_hr\" is supported in v0.1");
        }
        if !(s.cfl > 0.0 && s.cfl <= 0.5) {
            return err("solver.cfl must be in (0, 0.5] (unsplit 2D positivity bound)");
        }
        if !(s.h_dry_m > 0.0 && s.h_eps_m >= s.h_dry_m) {
            return err("solver.h_dry_m must be > 0 and h_eps_m >= h_dry_m");
        }
        if s.screening_physics != "hll" && s.screening_physics != "local_inertial" {
            return err("solver.screening_physics must be \"hll\" or \"local_inertial\"");
        }
        if s.subdomain_replay != "off" && s.subdomain_replay != "auto" {
            return err("solver.subdomain_replay must be \"off\" or \"auto\"");
        }
        if s.subdomain_replay == "auto" && s.dt_mode != DtMode::BaselineLocked {
            return err("solver.subdomain_replay = \"auto\" requires dt_mode = \"baseline_locked\"");
        }
        if let Some(sh) = &self.optimizer.level_budget_share
            && sh.len() != self.optimizer.fidelity_levels.len() {
                return err("optimizer.level_budget_share must match fidelity_levels length");
            }
        if self.optimizer.fidelity_levels.last() != Some(&1) {
            return err("optimizer.fidelity_levels must end with 1 (final ranking at full resolution)");
        }
        if !["cma_es", "lq_cma_es", "random", "sobol", "coordinate"].contains(&self.optimizer.method.as_str()) {
            return err("optimizer.method must be cma_es, lq_cma_es, random, sobol or coordinate");
        }
        if let Some(r) = &self.robustness {
            if r.members.is_empty() || !(0.0..1.0).contains(&r.alpha) || r.beta < 0.0 {
                return err("robustness: need ≥ 1 member, alpha in [0,1), beta ≥ 0");
            }
            if r.members.iter().any(|m| !(m.rain_scale >= 0.0 && m.inflow_scale >= 0.0)) {
                return err("robustness: member scales must be ≥ 0");
            }
        }
        if let Some(d) = &self.design
            && d.placement != "free" && d.placement != "corridor" {
                return err("design.placement must be \"free\" or \"corridor\"");
            }
        if let Some(o) = &self.objectives
            && o.building_mode != "ground" && o.building_mode != "wall" {
                return err("objectives.building_mode must be \"ground\" or \"wall\"");
            }
        Ok(())
    }

    /// Canonical JSON of the problem-defining part (hashed into the problem hash).
    pub fn problem_json(&self) -> String {
        serde_json::json!({
            "terrain": self.terrain, "hydrology": self.hydrology, "boundaries": self.boundaries,
            "design": self.design, "objectives": self.objectives, "robustness": self.robustness,
        })
        .to_string()
    }
    pub fn solver_json(&self) -> String {
        serde_json::to_string(&self.solver).unwrap()
    }
}

/// Unit-normalized forcing ready for the solver.
#[derive(Clone, Debug)]
pub struct Hyetograph {
    /// (t [s], rate [m/s]) piecewise-linear; constant extrapolation outside.
    pub pts: Vec<(f64, f64)>,
}

impl Hyetograph {
    pub fn from_mm_h(p: &[[f64; 2]]) -> Self {
        Self { pts: p.iter().map(|q| (q[0], q[1] / 1000.0 / 3600.0)).collect() }
    }
    pub fn from_pairs(p: &[[f64; 2]]) -> Self {
        Self { pts: p.iter().map(|q| (q[0], q[1])).collect() }
    }
    pub fn rate(&self, t: f64) -> f64 {
        let p = &self.pts;
        if p.is_empty() {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            if t <= w[1].0 {
                let (t0, r0) = w[0];
                let (t1, r1) = w[1];
                if t1 == t0 {
                    return r1;
                }
                return r0 + (r1 - r0) * (t - t0) / (t1 - t0);
            }
        }
        p[p.len() - 1].1
    }
    /// Exact integral of the piecewise-linear rate over `[a, b]` (spec §4.3).
    pub fn integral(&self, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let p = &self.pts;
        if p.is_empty() {
            return 0.0;
        }
        // Breakpoints inside (a,b) split the integral into trapezoids, which are exact
        // for piecewise-linear functions.
        let mut total = 0.0;
        let mut t = a;
        let mut r = self.rate(a);
        for &(tk, _) in p.iter() {
            if tk > a && tk < b {
                let rk = self.rate(tk);
                total += 0.5 * (r + rk) * (tk - t);
                t = tk;
                r = rk;
            }
        }
        let rb = self.rate(b);
        total += 0.5 * (r + rb) * (b - t);
        total
    }
    pub fn is_zero(&self) -> bool {
        self.pts.iter().all(|p| p.1 == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EX: &str = r#"
schema_version = "0.2"
scenario_id = "t"
[terrain]
path = "dem.tif"
crs = "EPSG:32643"
vertical_datum = "local"
[hydrology]
duration_s = 100
sync_interval_s = 10
rainfall_hyetograph = [[0, 0], [10, 36], [20, 0]]
manning_n = 0.03
[hydrology.infiltration]
model = "constant_capacity"
capacity_mm_h = 5.0
"#;

    #[test]
    fn parses_and_rejects_unknown() {
        Scenario::from_toml(EX).unwrap();
        let bad = EX.replace("manning_n = 0.03", "manning_n = 0.03\nbogus = 1");
        assert!(Scenario::from_toml(&bad).is_err());
    }

    #[test]
    fn hyetograph_integral_exact() {
        let h = Hyetograph::from_mm_h(&[[0.0, 0.0], [10.0, 36.0], [20.0, 0.0]]);
        // triangle: peak 36 mm/h = 1e-5 m/s, base 20 s → 1e-4 m
        let total = h.integral(0.0, 30.0);
        assert!((total - 1e-4).abs() < 1e-18);
        let split: f64 = (0..30).map(|k| h.integral(k as f64 * 0.7, (k + 1) as f64 * 0.7)).sum::<f64>()
            + h.integral(21.0, 30.0);
        assert!((split - total).abs() < 1e-15);
    }
}
