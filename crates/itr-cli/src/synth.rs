//! Synthetic example generators (spec §12 Phase A, §9.2): each writes a DEM GeoTIFF,
//! GeoJSON masks, an earthworks file and a scenario into a directory.

use crate::io::{self, polys_to_geojson, Res};
use itr_core::raster::{GeoTransform, Raster};
use serde_json::json;
use std::path::Path;

const X0: f64 = 500_000.0;
const Y0: f64 = 4_200_000.0;

type Ring = Vec<(f64, f64)>;

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Ring {
    // Local metres (from the south-west corner) → world.
    vec![(X0 + x0, Y0 + y0), (X0 + x1, Y0 + y0), (X0 + x1, Y0 + y1), (X0 + x0, Y0 + y1), (X0 + x0, Y0 + y0)]
}

fn dem(nx: usize, ny: usize, dx: f64, z: impl Fn(f64, f64) -> f64) -> Raster {
    let geo = GeoTransform { origin_x: X0, origin_y: Y0 + ny as f64 * dx, dx, dy: dx };
    let mut r = Raster::new(nx, ny, geo.clone(), 0.0);
    for row in 0..ny {
        for col in 0..nx {
            let (x, y) = geo.cell_center(col, row);
            r.set(col, row, z(x - X0, y - Y0));
        }
    }
    r
}

fn write(dir: &Path, name: &str, s: &str) -> Res<()> {
    std::fs::write(dir.join(name), s).map_err(|e| e.to_string())
}

fn gj(dir: &Path, name: &str, feats: Vec<(Ring, serde_json::Value)>) -> Res<()> {
    io::write_json(&dir.join(name), &polys_to_geojson(&feats))
}

const PRIMS: &str = r#"# Earthwork primitives (spec §6.1). Bounds are [min, max]; centres default to the
# editable mask's bounding box.
[[primitive]]
kind = "berm"
height_m = [0.0, 1.0]
width_m = [4.0, 12.0]
length_m = [10.0, 90.0]
angle_deg = [0.0, 180.0]

[[primitive]]
kind = "berm"
height_m = [0.0, 1.0]
width_m = [4.0, 12.0]
length_m = [10.0, 90.0]
angle_deg = [0.0, 180.0]

[[primitive]]
kind = "swale"
height_m = [0.0, 0.6]
width_m = [4.0, 10.0]
length_m = [10.0, 90.0]
angle_deg = [0.0, 180.0]
"#;

/// A meandering valley: inflow hydrograph + rain, houses on the north floodplain,
/// a guarded neighbourhood on the south floodplain downstream.
pub fn synthetic_valley(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let (nx, ny, dx) = (100, 80, 5.0);
    let yc = |x: f64| 200.0 + 30.0 * (x / 80.0).sin();
    let r = dem(nx, ny, dx, |x, y| {
        let d = y - yc(x);
        50.0 - 0.004 * x + 0.006 * d.abs() - 0.5 * (-(d / 8.0) * (d / 8.0)).exp()
    });
    io::write_geotiff(&dir.join("dem.tif"), &r, Some(32643))?;
    gj(dir, "protected.geojson", vec![
        (rect(320.0, 188.0, 345.0, 203.0), json!({"name": "house_a", "threshold_m": 0.10})),
        (rect(355.0, 183.0, 380.0, 198.0), json!({"name": "house_b", "threshold_m": 0.10})),
        (rect(390.0, 185.0, 412.0, 200.0), json!({"name": "house_c", "threshold_m": 0.10})),
    ])?;
    gj(dir, "guard.geojson", vec![(rect(300.0, 95.0, 460.0, 155.0), json!({"name": "south_neighbourhood"}))])?;
    gj(dir, "editable.geojson", vec![(rect(180.0, 150.0, 315.0, 290.0), json!({"name": "north_fields"}))])?;
    write(dir, "earthworks.toml", PRIMS)?;
    write(dir, "scenario.toml", r#"schema_version = "0.2"
scenario_id = "synthetic-valley-001"

[terrain]
path = "dem.tif"
crs = "EPSG:32643"
vertical_datum = "synthetic-local"
elevation_units = "m"
expected_cell_size_m = 5.0

[hydrology]
duration_s = 2400
sync_interval_s = 60
rainfall_hyetograph = [[0, 0], [300, 30], [1200, 10], [1800, 0]]
manning_n = 0.04

[hydrology.infiltration]
model = "constant_capacity"
capacity_mm_h = 5.0

[boundaries]
default = "wall"
segments = [
  { kind = "inflow", side = "west", hydrograph = [[0, 0], [600, 20], [1500, 6], [2400, 1]], range_m = [170, 230] },
  { kind = "transmissive", side = "east" },
]

[design]
editable_mask = "editable.geojson"
primitives = "earthworks.toml"
max_abs_elevation_change_m = 1.0
max_earthwork_volume_m3 = 2500
max_edit_slope = 0.5
require_no_offsite_worsening = true

[objectives]
protected_areas = "protected.geojson"
downstream_guard_areas = "guard.geojson"
depth_threshold_m = 0.10
guard_tolerance_m = 0.03
earthwork_weight = 1e-5

[solver]
precision = "f32"
cfl = 0.35

[optimizer]
method = "cma_es"
seed = 12345
max_simulations = 120
fidelity_levels = [2, 1]
level_budget_share = [0.6, 0.4]
"#)
}

/// Rain on a hillside above a village. Diverting water east floods the guarded
/// neighbour (must be rejected); diverting west into the pond is the feasible fix.
pub fn diverted_flood(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let (nx, ny, dx) = (90, 70, 5.0);
    let r = dem(nx, ny, dx, |x, y| {
        let pond = -1.5 * (-(((x - 115.0) / 45.0).powi(2) + ((y - 85.0) / 30.0).powi(2))).exp();
        let swale_to_village = -0.5 * (-((x - 220.0) / 35.0).powi(2)).exp();
        30.0 + 0.03 * y + pond + swale_to_village + 0.05 * (x / 23.0).sin()
    });
    io::write_geotiff(&dir.join("dem.tif"), &r, Some(32643))?;
    gj(dir, "protected.geojson", vec![
        (rect(190.0, 30.0, 215.0, 50.0), json!({"name": "village_w", "threshold_m": 0.05})),
        (rect(225.0, 30.0, 250.0, 50.0), json!({"name": "village_e", "threshold_m": 0.05})),
    ])?;
    gj(dir, "guard.geojson", vec![(rect(300.0, 20.0, 420.0, 90.0), json!({"name": "neighbour"}))])?;
    gj(dir, "editable.geojson", vec![(rect(60.0, 60.0, 400.0, 200.0), json!({"name": "hillside"}))])?;
    write(dir, "earthworks.toml", r#"[[primitive]]
kind = "berm"
height_m = [0.0, 1.0]
width_m = [4.0, 10.0]
length_m = [20.0, 160.0]
angle_deg = [0.0, 180.0]

[[primitive]]
kind = "swale"
height_m = [0.0, 0.6]
width_m = [4.0, 10.0]
length_m = [20.0, 160.0]
angle_deg = [0.0, 180.0]
"#)?;
    write(dir, "scenario.toml", r#"schema_version = "0.2"
scenario_id = "diverted-flood-001"

[terrain]
path = "dem.tif"
crs = "EPSG:32643"
vertical_datum = "synthetic-local"

[hydrology]
duration_s = 2400
sync_interval_s = 60
rainfall_hyetograph = [[0, 0], [120, 80], [1500, 40], [1800, 0]]
manning_n = 0.05

[hydrology.infiltration]
model = "constant_capacity"
capacity_mm_h = 10.0

[boundaries]
segments = [{ kind = "transmissive", side = "south" }]

[design]
editable_mask = "editable.geojson"
primitives = "earthworks.toml"
max_abs_elevation_change_m = 1.0
max_earthwork_volume_m3 = 3000
# Primitives are placed along flow corridors traced upstream from the assets (§7.11.4);
# 20% of each population still samples free placement.
placement = "corridor"

[objectives]
protected_areas = "protected.geojson"
downstream_guard_areas = "guard.geojson"
depth_threshold_m = 0.05
guard_tolerance_m = 0.01

[optimizer]
seed = 7
max_simulations = 200
fidelity_levels = [1]
"#)
}

/// Houses at the bottom of a closed bowl under heavy rain, with the whole map guarded.
/// No permitted earthwork can lower their peak depth without worsening elsewhere: the
/// engine must report *infeasible*, not a fabricated success.
pub fn infeasible_problem(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let (nx, ny, dx) = (64, 64, 5.0);
    let r = dem(nx, ny, dx, |x, y| {
        let r2 = (x - 160.0).powi(2) + (y - 160.0).powi(2);
        20.0 + 0.0002 * r2
    });
    io::write_geotiff(&dir.join("dem.tif"), &r, Some(32643))?;
    gj(dir, "protected.geojson", vec![(rect(145.0, 145.0, 175.0, 175.0), json!({"name": "bowl_houses", "threshold_m": 0.05}))])?;
    gj(dir, "editable.geojson", vec![(rect(240.0, 40.0, 300.0, 280.0), json!({"name": "east_rim"}))])?;
    write(dir, "earthworks.toml", r#"[[primitive]]
kind = "berm"
height_m = [0.0, 0.5]
width_m = [4.0, 8.0]
length_m = [10.0, 40.0]

[[primitive]]
kind = "mound"
height_m = [0.0, 0.5]
width_m = [6.0, 20.0]
"#)?;
    write(dir, "scenario.toml", r#"schema_version = "0.2"
scenario_id = "infeasible-problem-001"

[terrain]
path = "dem.tif"
crs = "EPSG:32643"
vertical_datum = "synthetic-local"

[hydrology]
duration_s = 1800
sync_interval_s = 60
rainfall_hyetograph = [[0, 50], [1200, 50], [1260, 0]]
manning_n = 0.04

[hydrology.infiltration]
model = "none"

[design]
editable_mask = "editable.geojson"
primitives = "earthworks.toml"
max_abs_elevation_change_m = 0.5
max_earthwork_volume_m3 = 300

[objectives]
protected_areas = "protected.geojson"
depth_threshold_m = 0.05
guard_tolerance_m = 0.005
guard_whole_map = true

[optimizer]
seed = 3
max_simulations = 40
fidelity_levels = [1]
"#)
}

pub fn all(root: &Path) -> Res<()> {
    synthetic_valley(&root.join("synthetic-valley"))?;
    diverted_flood(&root.join("diverted-flood"))?;
    infeasible_problem(&root.join("infeasible-problem"))
}
