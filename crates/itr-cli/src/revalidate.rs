//! Post-search revalidation (spec §9.2 "model exploitation", Phase D): the final design is
//! re-run against the no-change terrain on a 2× finer grid and under alternate rainfall
//! events. A case where the design no longer beats no-change, or violates a guard, is a
//! *reversal* and is reported, never hidden.

use crate::io::Res;
use crate::run;
use itr_core::design::{Primitive, TerrainEdit};
use itr_core::raster::{GeoTransform, Mask, Raster};
use itr_core::Real;
use itr_opt::{Problem, ProblemInputs};
use serde_json::{json, Value};

/// Refined grids above this many cells are skipped (reported, not silently dropped).
const MAX_REFINED_CELLS: usize = 4_000_000;

/// Upsample by `f`: bilinear DEM between cell centres (nearest where a neighbour is
/// nodata), nearest-parent for masks and Manning's n.
pub fn refine(inp: &ProblemInputs, f: usize) -> ProblemInputs {
    let (nx, ny, g) = (inp.dem.nx, inp.dem.ny, &inp.dem.geo);
    let (mx, my) = (nx * f, ny * f);
    let geo = GeoTransform { origin_x: g.origin_x, origin_y: g.origin_y, dx: g.dx / f as f64, dy: g.dy / f as f64 };
    let parent = |k: usize| (k / mx / f) * nx + (k % mx) / f;
    let mut dem = Raster::new(mx, my, geo, f64::NAN);
    for r in 0..my {
        // Fine-cell centre in coarse cell-centre coordinates.
        let y = ((r as f64 + 0.5) / f as f64 - 0.5).clamp(0.0, (ny - 1) as f64);
        let (r0, wy) = (y.floor() as usize, y.fract());
        let r1 = (r0 + 1).min(ny - 1);
        for c in 0..mx {
            let x = ((c as f64 + 0.5) / f as f64 - 0.5).clamp(0.0, (nx - 1) as f64);
            let (c0, wx) = (x.floor() as usize, x.fract());
            let c1 = (c0 + 1).min(nx - 1);
            let z = |cc, rr| inp.dem.get(cc, rr);
            let v = (1.0 - wy) * ((1.0 - wx) * z(c0, r0) + wx * z(c1, r0)) + wy * ((1.0 - wx) * z(c0, r1) + wx * z(c1, r1));
            dem.data[r * mx + c] = if v.is_finite() { v } else { inp.dem.data[parent(r * mx + c)] };
        }
    }
    let up = |m: &Mask| Mask { nx: mx, ny: my, data: (0..mx * my).map(|k| m.data[parent(k)]).collect() };
    ProblemInputs {
        scenario: inp.scenario.clone(),
        dem,
        walls: up(&inp.walls),
        manning: (0..mx * my).map(|k| inp.manning[parent(k)]).collect(),
        editable: inp.editable.as_ref().map(up),
        assets: inp.assets.iter().map(|(n, m, t, w)| (n.clone(), up(m), *t, *w)).collect(),
        guard: inp.guard.as_ref().map(up),
        primitives: inp.primitives.clone(),
    }
}

/// Run no-change and the design on `inputs` at full resolution and compare.
fn case<T: Real>(name: &str, what: &str, mut inputs: ProblemInputs, prims: &[Primitive], threads: usize) -> Res<Value> {
    inputs.scenario.optimizer.fidelity_levels = vec![1];
    inputs.scenario.optimizer.level_budget_share = None;
    // Revalidation is per event; the robust ensemble is not re-run here.
    inputs.scenario.robustness = None;
    let (nx, ny, geo) = (inputs.dem.nx, inputs.dem.ny, inputs.dem.geo.clone());
    let pb = Problem::<T>::build(inputs, threads, 0)?;
    let obj = run::full_objective(&pb);
    let base = run::full_run(&pb, &obj, &TerrainEdit::default(), None, threads)?;
    let ds = pb.design.as_ref().ok_or("revalidation needs a design space")?;
    let edit = ds.materialize(prims, &geo, nx, ny, &pb.editable);
    let cand = run::full_run(&pb, &obj, &edit, Some(&base.summary.monitor_max), threads)?;
    let (b, c) = (&base.report, &cand.report);
    let holds = c.feasible && c.j < b.j;
    Ok(json!({
        "case": name, "description": what, "grid": [nx, ny], "cell_size_m": geo.dx,
        "j_before": b.j, "j_after": c.j, "relative_change": (c.j - b.j) / b.j.abs().max(1e-12),
        "feasible": c.feasible, "guard_max_worsening_m": c.guard_max_worsening_m,
        "assets_exceeding_before": b.assets_exceeding, "assets_exceeding_after": c.assets_exceeding,
        "cut_m3": c.cut_m3, "fill_m3": c.fill_m3,
        "mass_rel_residual": [base.summary.ledger.rel_residual, cand.summary.ledger.rel_residual],
        "reversal": !holds,
    }))
}

/// Nominal design check on finer grid and alternate rainfall. Rain scales multiply every
/// hyetograph rate (inflow hydrographs are unchanged).
pub fn revalidate<T: Real>(inputs: &ProblemInputs, prims: &[Primitive], threads: usize) -> Res<Value> {
    let mut cases = vec![];
    let n = inputs.dem.nx * inputs.dem.ny * 4;
    if n <= MAX_REFINED_CELLS {
        cases.push(case::<T>("finer_grid_2x", "DEM bilinearly refined to half the cell size; same event", refine(inputs, 2), prims, threads)?);
    } else {
        cases.push(json!({"case": "finer_grid_2x", "skipped": format!("{n} cells exceeds the {MAX_REFINED_CELLS}-cell revalidation limit")}));
    }
    for (name, s) in [("rain_x0.5", 0.5), ("rain_x1.5", 1.5)] {
        let mut inp = inputs.clone();
        for p in inp.scenario.hydrology.rainfall_hyetograph.iter_mut() {
            p[1] *= s;
        }
        cases.push(case::<T>(name, &format!("rainfall rates × {s}; native grid"), inp, prims, threads)?);
    }
    let reversals: Vec<&str> = cases.iter().filter(|c| c["reversal"] == true).filter_map(|c| c["case"].as_str()).collect();
    Ok(json!({
        "method": "spec §9.2: final design re-run against no-change on a 2× finer grid and alternate rainfall; a reversal is any case where the design is infeasible or does not lower J",
        "reversals": reversals,
        "cases": cases,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use itr_core::scenario::Scenario;

    #[test]
    fn refine_preserves_planes_and_masks() {
        let geo = GeoTransform { origin_x: 100.0, origin_y: 200.0, dx: 4.0, dy: 4.0 };
        let mut dem = Raster::new(5, 4, geo, 0.0);
        for r in 0..4 {
            for c in 0..5 {
                dem.set(c, r, 2.0 * c as f64 - 3.0 * r as f64);
            }
        }
        let mut m = Mask::new(5, 4, false);
        m.set(2, 1, true);
        let sc = Scenario::from_toml(include_str!("../../../examples/synthetic-valley/scenario.toml")).unwrap();
        let inp = ProblemInputs { scenario: sc, dem, walls: m.clone(), manning: vec![0.03; 20], editable: None, assets: vec![], guard: None, primitives: vec![] };
        let f = refine(&inp, 2);
        assert_eq!((f.dem.nx, f.dem.ny, f.dem.geo.dx), (10, 8, 2.0));
        // Interior fine cell (3, 3) has centre at coarse coordinates (1.25, 1.25).
        assert!((f.dem.get(3, 3) - (2.0 * 1.25 - 3.0 * 1.25)).abs() < 1e-12);
        assert_eq!(f.walls.count(), 4);
        assert!(f.walls.get(4, 2) && f.walls.get(5, 3));
    }
}
