# Scenario schema (v0.2)

Scenarios are TOML; JSON with the same structure is also accepted. Unknown keys are rejected. Paths are relative to the scenario file. The authoritative definition is `crates/itr-core/src/scenario.rs`.

```toml
schema_version = "0.2"
scenario_id = "synthetic-valley-001"

[terrain]
path = "dem.tif"                 # projected single-band GeoTIFF; nodata → solid wall
crs = "EPSG:32643"               # must match the GeoTIFF's ProjectedCSType if declared
vertical_datum = "local"
elevation_units = "m"            # only "m"
expected_cell_size_m = 5.0       # optional check
manning_path = "n.tif"           # optional Manning-n raster on the DEM grid
obstacles = "walls.geojson"      # optional solid obstacles

[hydrology]
duration_s = 2400
sync_interval_s = 60             # Δt is clipped to land on these instants; part of the problem hash
rainfall_hyetograph = [[0, 0], [300, 30], [1200, 10], [1800, 0]]   # [t_s, mm/h], piecewise-linear
manning_n = 0.04
initial_depth_m = 0.0
initial_stage_m = 101.5          # optional: lake surface elevation where above ground

[hydrology.infiltration]
model = "constant_capacity"      # or "none"
capacity_mm_h = 5.0

[boundaries]
default = "wall"
segments = [
  # range_m: metres along the side from its first cell
  # (the north edge for west/east sides, the west edge for north/south sides)
  { kind = "inflow", side = "west", hydrograph = [[0, 0], [600, 20]], range_m = [170, 230] },   # Q in m³/s
  { kind = "stage", side = "south", stage_m = 47.5 },
  { kind = "transmissive", side = "east" },   # partially reflective; reported as such
]

[design]
editable_mask = "editable.geojson"   # or a .tif on the DEM grid (non-zero = editable)
primitives = "earthworks.toml"
max_abs_elevation_change_m = 1.0
max_earthwork_volume_m3 = 2500       # cut + fill
max_edit_slope = 0.5                 # |∇Δz|; 0 disables
require_no_offsite_worsening = true
placement = "free"                   # "corridor": place along flow corridors traced upstream from the assets
free_placement_fraction = 0.2        # corridor mode: share of each population sampled with free placement
cut_cost_per_m3 = 1.0
fill_cost_per_m3 = 1.0
quantize = { height_m = 0.01, position_m = 0.5, width_m = 0.5, angle_deg = 1.0 }

[objectives]
protected_areas = "protected.geojson"   # one asset per feature; properties name, threshold_m, weight
downstream_guard_areas = "guard.geojson"
depth_threshold_m = 0.10                # default asset threshold
guard_tolerance_m = 0.03
building_mode = "ground"                # "wall": footprints become solid; their 1-cell ring is monitored
smoothing_m = 0.02                      # softplus τ
earthwork_weight = 1e-5                 # λ in J = J_risk + λ·J_earth (disclosed in metrics.json)
guard_whole_map = false                 # guard every non-protected cell
monitor_full_domain = false

[solver]
precision = "f32"        # "f64" = oracle precision
backend = "cpu"          # "gpu": f32 batches on wgpu (build with --features gpu); unsupported levels use the CPU
dt_mode = "adaptive"     # "baseline_locked": reuse the baseline Δt schedule (hard-CFL checked, adaptive fallback)
subdomain_replay = "off" # "auto": simulate candidates on an ROI around the editable area (needs baseline_locked)
screening_physics = "hll"  # "local_inertial": cheaper physics on coarse levels (level 1 is always HLL)
cfl = 0.35               # (0, 0.5]
dt_max_s = 5.0
h_dry_m = 1e-6
h_eps_m = 1e-5

[optimizer]
method = "cma_es"        # "cma_es" | "lq_cma_es" | "random" | "sobol" | "coordinate"
seed = 12345
population = 12          # default 4 + ⌊3 ln n⌋; never depends on core count
max_simulations = 120
max_cell_updates = 10000000000  # optional second budget (integer)
fidelity_levels = [2, 1] # coarsening factors, coarse → fine; must end with 1
level_budget_share = [0.6, 0.4]
initial_sigma = 0.3
max_static_attempts = 50
```

Optional robust objective over an ensemble (§6.6): `J = E[J] + β·CVaR_α(J)`, feasible only if every member is.

```toml
[robustness]
beta = 0.5
alpha = 0.8                      # CVaR over the worst (1 − α) share of members
uncertainty_model = "±30% design-storm depth"   # recorded in the manifest
members = [
  { name = "nominal", rain_scale = 1.0, inflow_scale = 1.0 },   # member 0: full-output rerun
  { name = "wet", rain_scale = 1.3 },
  { name = "dry", rain_scale = 0.7 },
]
```

## `itr serve-eval` protocol

Newline-delimited JSON on stdin/stdout (log lines go to stderr). Points are in the unit box `[0,1]^n`.

```text
→ {"id": 1, "cmd": "info"}
← {"id": 1, "dimension": 8, "param_names": [...], "levels": [1], "baseline_j": 0.0279, ...}
→ {"id": 2, "cmd": "eval", "x": [[0.1, 0.5, ...], ...], "level": 1, "incumbent": null}
← {"id": 2, "results": [{"status": "ok", "feasible": true, "j": 0.012, "violation": 0, ...}], ...}
→ {"id": 3, "cmd": "quit"}
```

## Primitives file

```toml
[[primitive]]
kind = "berm"            # berm | swale | cut | mound
height_m = [0.0, 1.0]    # magnitude; the sign comes from the kind
width_m = [4.0, 12.0]
length_m = [10.0, 90.0]  # linear kinds only
angle_deg = [0.0, 180.0] # linear kinds only
x_m = [500100, 500300]   # optional centre bounds (default: editable bbox)
y_m = [4200150, 4200290]
```

Linear kinds have 6 parameters `(x, y, length, width, angle, height)`; a mound has 4 `(x, y, width, height)`.
