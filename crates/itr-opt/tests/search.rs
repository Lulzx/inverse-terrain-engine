//! Optimization tests (spec §9.2) on a small in-memory problem.

use itr_core::design::{PrimitiveKind, PrimitiveSpec};
use itr_core::model::NoopObserver;
use itr_core::raster::{GeoTransform, Mask, Raster};
use itr_core::scenario::Scenario;
use itr_hydro::PreparedTerrain;
use itr_opt::evaluator::Status;
use itr_opt::{search, Evaluator, Problem, ProblemInputs, SearchEntry};

const SC: &str = r#"
schema_version = "0.2"
scenario_id = "test"
[terrain]
path = "mem"
crs = "EPSG:32643"
vertical_datum = "local"
[hydrology]
duration_s = 900
sync_interval_s = 60
rainfall_hyetograph = [[0, 0], [120, 0], [180, 90], [600, 0]]
manning_n = 0.04
[hydrology.infiltration]
model = "none"
[boundaries]
segments = [{ kind = "transmissive", side = "south" }]
[design]
editable_mask = "mem"
primitives = "mem"
max_abs_elevation_change_m = 0.8
max_earthwork_volume_m3 = 5000
[objectives]
protected_areas = "mem"
downstream_guard_areas = "mem"
depth_threshold_m = 0.03
guard_tolerance_m = 0.005
[optimizer]
seed = 11
max_simulations = 24
population = 6
fidelity_levels = [2, 1]
level_budget_share = [0.5, 0.5]
"#;

fn inputs() -> ProblemInputs {
    let (nx, ny, dx) = (40, 32, 4.0);
    let geo = GeoTransform { origin_x: 0.0, origin_y: ny as f64 * dx, dx, dy: dx };
    let mut dem = Raster::new(nx, ny, geo.clone(), 0.0);
    for r in 0..ny {
        for c in 0..nx {
            let (x, y) = geo.cell_center(c, r);
            dem.set(c, r, 10.0 + 0.02 * y - 0.3 * (-((x - 80.0) / 20.0f64).powi(2)).exp() + 0.03 * (x / 9.0).sin());
        }
    }
    let rect = |c0: usize, r0: usize, c1: usize, r1: usize| {
        let mut m = Mask::new(nx, ny, false);
        for r in r0..r1 {
            for c in c0..c1 {
                m.set(c, r, true);
            }
        }
        m
    };
    let spec = |kind| PrimitiveSpec {
        kind,
        height_m: [0.0, 0.6],
        width_m: [8.0, 16.0],
        length_m: [10.0, 60.0],
        angle_deg: [0.0, 180.0],
        x_m: None,
        y_m: None,
    };
    ProblemInputs {
        scenario: Scenario::from_toml(SC).unwrap(),
        dem,
        walls: Mask::new(nx, ny, false),
        manning: vec![0.04; nx * ny],
        editable: Some(rect(5, 4, 36, 18)),
        assets: vec![("house".into(), rect(17, 26, 23, 30), None, 1.0)],
        guard: Some(rect(28, 22, 38, 31)),
        primitives: vec![spec(PrimitiveKind::Berm), spec(PrimitiveKind::Swale)],
    }
}

fn strip(v: &[SearchEntry]) -> Vec<String> {
    v.iter()
        .map(|e| {
            let mut e = e.clone();
            e.record.wall_s = 0.0;
            serde_json::to_string(&e).unwrap()
        })
        .collect()
}

struct Collect(Vec<SearchEntry>);
impl itr_opt::Sink for Collect {
    fn on_eval(&mut self, e: &SearchEntry) {
        self.0.push(e.clone());
    }
}

#[test]
fn trace_independent_of_workers_and_resume_is_exact() {
    let pb = Problem::<f32>::build(inputs(), 1, 64 << 20).unwrap();
    let mut a = Collect(vec![]);
    let ra = search(&pb, &mut Evaluator::new(&pb, 1), &mut a);
    let mut b = Collect(vec![]);
    let mut evb = Evaluator::new(&pb, 3);
    let _ = search(&pb, &mut evb, &mut b);
    assert_eq!(strip(&a.0), strip(&b.0), "trace depends on worker count");
    assert!(ra.sims >= 24);
    // Resume: prefill the cache with the first half of the fresh evaluations.
    let mut evc = Evaluator::new(&pb, 2);
    for (k, r) in evb.cache.values().enumerate() {
        if k % 2 == 0 {
            evc.cache.insert(r.key.clone(), r.clone());
        }
    }
    let mut c = Collect(vec![]);
    let _ = search(&pb, &mut evc, &mut c);
    assert_eq!(strip(&a.0), strip(&c.0), "resumed trace differs");
}

#[test]
fn zero_intervention_is_bitwise_baseline() {
    let pb = Problem::<f32>::build(inputs(), 1, 64 << 20).unwrap();
    let n = pb.design.as_ref().unwrap().dimension();
    // Height is the last parameter of each linear primitive; u = 0 → height 0.
    let mut u = vec![0.5; n];
    u[5] = 0.0;
    u[11] = 0.0;
    let li = pb.levels.len() - 1;
    let c = Evaluator::prepare(&pb, li, &u);
    assert!(c.edit.is_empty());
    let mut ev = Evaluator::new(&pb, 1);
    let r = ev.eval_batch(&pb, li, &[c], None).remove(0);
    let lv = &pb.levels[li];
    let base = lv.objective.evaluate(&lv.baseline_max, &Default::default(), Some(&lv.baseline_max));
    assert_eq!(r.j.to_bits(), base.j.to_bits());
    assert_eq!(r.steps, lv.baseline.steps);
    assert_eq!(r.status, Status::Ok);
}

#[test]
fn guard_abort_is_sound() {
    // Every guard-aborted candidate, re-run to completion, is infeasible (§7.6, §9.2).
    let mut inp = inputs();
    inp.scenario.optimizer.method = "random".into();
    inp.scenario.optimizer.max_simulations = 60;
    // Whole map outside the house is guarded: most edits move water somewhere.
    inp.scenario.objectives.as_mut().unwrap().guard_whole_map = true;
    inp.scenario.optimizer.fidelity_levels = vec![1];
    inp.scenario.optimizer.level_budget_share = None;
    let pb = Problem::<f64>::build(inp, 1, 64 << 20).unwrap();
    let mut col = Collect(vec![]);
    let _ = search(&pb, &mut Evaluator::new(&pb, 2), &mut col);
    let aborted: Vec<&SearchEntry> = col.0.iter().filter(|e| e.record.status == Status::GuardAbort).collect();
    assert!(!aborted.is_empty(), "test problem produced no guard aborts");
    let lv = &pb.levels[0];
    let ds = pb.design.as_ref().unwrap();
    let d = &pb.inputs.dem;
    for e in aborted {
        let edit = ds.materialize(&e.record.primitives, &d.geo, d.nx, d.ny, &pb.editable);
        let mut terr = PreparedTerrain::new(lv.base.clone());
        terr.apply_edit(&edit);
        let mut ws = itr_hydro::Workspace::new(lv.base.layout, &lv.objective.monitors, 1, 32);
        let s = pb.solver().run_impl(&mut ws, &terr, None, &mut NoopObserver).unwrap();
        let rep = lv.objective.evaluate(&s.monitor_max, &edit, Some(&lv.baseline_max));
        assert!(!rep.feasible, "guard-aborted candidate {} is feasible when run to completion", e.eval_index);
    }
}

#[test]
fn robust_objective_over_ensemble() {
    use itr_core::scenario::{EnsembleMember, RobustnessCfg};
    let mut inp = inputs();
    inp.scenario.optimizer.fidelity_levels = vec![1];
    inp.scenario.optimizer.level_budget_share = None;
    inp.scenario.optimizer.max_simulations = 12;
    inp.scenario.robustness = Some(RobustnessCfg {
        beta: 0.5,
        alpha: 0.5,
        members: vec![
            EnsembleMember { name: "nominal".into(), rain_scale: 1.0, inflow_scale: 1.0 },
            EnsembleMember { name: "wet".into(), rain_scale: 1.4, inflow_scale: 1.0 },
        ],
        uncertainty_model: "test".into(),
    });
    let pb = Problem::<f64>::build(inp, 1, 64 << 20).unwrap();
    assert_eq!(pb.levels[0].extra.len(), 1);
    // The wetter member floods the house more.
    let lv = &pb.levels[0];
    let base_nom = lv.objective.evaluate(&lv.baseline_max, &Default::default(), None).j;
    let base_wet = lv.objective.evaluate(&lv.extra[0].baseline_max, &Default::default(), None).j;
    assert!(base_wet > base_nom);
    let mut col = Collect(vec![]);
    let _ = search(&pb, &mut Evaluator::new(&pb, 2), &mut col);
    let full: Vec<&SearchEntry> = col.0.iter().filter(|e| e.record.members.len() == 2).collect();
    assert!(!full.is_empty());
    for e in full {
        let js: Vec<f64> = e.record.members.iter().map(|m| m.j).collect();
        let want = itr_opt::evaluator::robust_j(&js, 0.5, 0.5);
        assert_eq!(e.record.j.to_bits(), want.to_bits());
        // CVaR of the worse half = the wetter member here; J_robust = mean + β·max.
        assert!((want - (0.5 * (js[0] + js[1]) + 0.5 * js[0].max(js[1]))).abs() < 1e-12);
    }
}

#[test]
fn ablation_shapley_is_efficient() {
    // Exact Shapley values sum to the total improvement J(none) − J(all), and the
    // all-primitives subset reproduces the direct evaluation bitwise.
    let pb = Problem::<f64>::build(inputs(), 1, 64 << 20).unwrap();
    let li = pb.levels.len() - 1;
    let ds = pb.design.as_ref().unwrap();
    let u = vec![0.7; ds.dimension()];
    let mut ev = Evaluator::new(&pb, 2);
    let direct = ev.eval_batch(&pb, li, &[Evaluator::prepare(&pb, li, &u)], None).remove(0);
    let ab = itr_opt::ablation::ablate(&pb, &mut ev, li, &direct.primitives);
    assert!(ab.exact);
    assert_eq!(ab.subsets.len(), 1 << ab.attributions.len());
    assert_eq!(ab.j_all.to_bits(), direct.j.to_bits());
    let sum: f64 = ab.attributions.iter().map(|a| a.shapley_j_reduction.unwrap()).sum();
    assert!((sum - (ab.j_none - ab.j_all)).abs() < 1e-12, "{sum} vs {}", ab.j_none - ab.j_all);
}

#[test]
fn f32_ranking_agrees_with_f64() {
    // §9.1.1: f32 vs f64 metric differences within tolerance, top-k ranking agreement.
    let pb64 = Problem::<f64>::build(inputs(), 1, 64 << 20).unwrap();
    let pb32 = Problem::<f32>::build(inputs(), 1, 64 << 20).unwrap();
    let li = pb64.levels.len() - 1;
    let dim = pb64.design.as_ref().unwrap().dimension();
    let mut rng = itr_opt::rng::Rand::new(5);
    let us: Vec<Vec<f64>> = (0..16).map(|_| (0..dim).map(|_| rng.uniform()).collect()).collect();
    let j64: Vec<f64> = Evaluator::new(&pb64, 2).eval_batch(&pb64, li, &us.iter().map(|u| Evaluator::prepare(&pb64, li, u)).collect::<Vec<_>>(), None).iter().map(|r| r.j).collect();
    let j32: Vec<f64> = Evaluator::new(&pb32, 2).eval_batch(&pb32, li, &us.iter().map(|u| Evaluator::prepare(&pb32, li, u)).collect::<Vec<_>>(), None).iter().map(|r| r.j).collect();
    let (mut conc, mut disc) = (0, 0);
    for a in 0..us.len() {
        for b in a + 1..us.len() {
            let s = (j64[a] - j64[b]) * (j32[a] - j32[b]);
            if s > 0.0 {
                conc += 1;
            } else if s < 0.0 {
                disc += 1;
            }
        }
    }
    let tau = (conc - disc) as f64 / (conc + disc).max(1) as f64;
    let dmax = j64.iter().zip(&j32).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let best = |v: &[f64]| (0..v.len()).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    println!("f32 vs f64: Kendall tau {tau:.3}, max |ΔJ| {dmax:.2e}, best {} vs {}", best(&j64), best(&j32));
    assert!(tau > 0.9, "Kendall tau {tau}");
    assert!(dmax < 1e-3, "max |ΔJ| {dmax}");
    assert_eq!(best(&j64), best(&j32));
}
