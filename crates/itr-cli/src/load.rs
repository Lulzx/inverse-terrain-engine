//! Scenario loading: TOML + referenced rasters/vectors → `ProblemInputs`. Paths in a
//! scenario are relative to the scenario file. Content hashes go into the manifest.

use crate::io::{self, Res};
use itr_core::design::PrimitivesFile;
use itr_core::raster::{Mask, Raster};
use itr_core::scenario::Scenario;
use itr_opt::ProblemInputs;
use std::path::{Path, PathBuf};

pub struct Loaded {
    pub inputs: ProblemInputs,
    pub scenario_path: PathBuf,
    pub epsg: Option<u32>,
    /// (relative path, BLAKE3) of every input file.
    pub file_hashes: Vec<(String, String)>,
}

fn hash_file(p: &Path) -> String {
    std::fs::read(p).map(|b| itr_core::hash::hex(&b)).unwrap_or_default()
}

fn mask(path: &Path, dem: &Raster) -> Res<Mask> {
    if path.extension().is_some_and(|e| e == "tif" || e == "tiff") {
        let (r, _) = io::read_geotiff(path)?;
        if r.nx != dem.nx || r.ny != dem.ny || r.geo != dem.geo {
            return Err(format!("{}: mask raster must be on the DEM grid", path.display()));
        }
        let mut m = Mask::new(r.nx, r.ny, false);
        for (i, v) in r.data.iter().enumerate() {
            m.data[i] = v.is_finite() && *v != 0.0;
        }
        Ok(m)
    } else {
        io::mask_from_geojson(path, dem)
    }
}

pub fn load(scenario_path: &Path) -> Res<Loaded> {
    let text = std::fs::read_to_string(scenario_path).map_err(|e| format!("{}: {e}", scenario_path.display()))?;
    let sc = if scenario_path.extension().is_some_and(|e| e == "json") {
        let s: Scenario = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        s.validate().map_err(|e| e.0)?;
        s
    } else {
        Scenario::from_toml(&text).map_err(|e| format!("{}: {e}", scenario_path.display()))?
    };
    let dir = scenario_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let rel = |p: &str| dir.join(p);
    let mut file_hashes = vec![(scenario_path.display().to_string(), hash_file(scenario_path))];
    let mut track = |p: &Path| file_hashes.push((p.display().to_string(), hash_file(p)));

    let dem_path = rel(&sc.terrain.path);
    track(&dem_path);
    let (dem, info) = io::read_geotiff(&dem_path)?;
    if info.model_type == Some(2) {
        return Err(format!(
            "{}: geographic (lat/lon) CRS; reproject to a metric CRS first: gdalwarp -t_srs EPSG:<utm> in.tif out.tif",
            dem_path.display()
        ));
    }
    let want = io::epsg_of(&sc.terrain.crs);
    if let (Some(a), Some(b)) = (want, info.epsg)
        && a != b && b != 32767 {
            return Err(format!("CRS mismatch: scenario says EPSG:{a}, DEM declares EPSG:{b}"));
        }
    if let Some(cs) = sc.terrain.expected_cell_size_m
        && ((dem.geo.dx - cs).abs() > 1e-6 * cs || (dem.geo.dy - cs).abs() > 1e-6 * cs) {
            return Err(format!("cell size {}×{} m differs from expected_cell_size_m = {cs}", dem.geo.dx, dem.geo.dy));
        }
    let mut walls = Mask::new(dem.nx, dem.ny, false);
    for (i, v) in dem.data.iter().enumerate() {
        walls.data[i] = !v.is_finite();
    }
    if let Some(o) = &sc.terrain.obstacles {
        let p = rel(o);
        track(&p);
        walls.or(&mask(&p, &dem)?);
    }
    let manning = match &sc.terrain.manning_path {
        Some(mp) => {
            let p = rel(mp);
            track(&p);
            let (r, _) = io::read_geotiff(&p)?;
            if r.nx != dem.nx || r.ny != dem.ny {
                return Err("manning raster must be on the DEM grid".into());
            }
            r.data.iter().map(|v| if v.is_finite() { *v } else { sc.hydrology.manning_n }).collect()
        }
        None => vec![sc.hydrology.manning_n; dem.nx * dem.ny],
    };
    let (mut editable, mut primitives) = (None, vec![]);
    if let Some(d) = &sc.design {
        let p = rel(&d.editable_mask);
        track(&p);
        editable = Some(mask(&p, &dem)?);
        let pp = rel(&d.primitives);
        track(&pp);
        let t = std::fs::read_to_string(&pp).map_err(|e| format!("{}: {e}", pp.display()))?;
        primitives = PrimitivesFile::from_toml(&t).map_err(|e| format!("{}: {e}", pp.display()))?.primitive;
    }
    let (mut assets, mut guard) = (vec![], None);
    if let Some(o) = &sc.objectives {
        let p = rel(&o.protected_areas);
        track(&p);
        for (k, f) in io::read_geojson(&p)?.into_iter().enumerate() {
            let m = itr_core::raster::rasterize(&f.polys, dem.nx, dem.ny, &dem.geo);
            if m.count() == 0 {
                return Err(format!("{}: feature {k} covers no cell centre", p.display()));
            }
            let name = f.props["name"].as_str().map(String::from).unwrap_or(format!("asset_{k}"));
            assets.push((name, m, f.props["threshold_m"].as_f64(), f.props["weight"].as_f64().unwrap_or(1.0)));
        }
        if let Some(g) = &o.downstream_guard_areas {
            let p = rel(g);
            track(&p);
            guard = Some(mask(&p, &dem)?);
        }
    }
    Ok(Loaded {
        epsg: want.or(info.epsg),
        inputs: ProblemInputs { scenario: sc, dem, walls, manning, editable, assets, guard, primitives },
        scenario_path: scenario_path.to_path_buf(),
        file_hashes,
    })
}
