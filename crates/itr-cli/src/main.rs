//! `itr` — Inverse Terrain Engine command line (spec §8.1).

mod backend;
mod cad;
mod io;
mod load;
mod run;
mod serve;
mod synth;
mod viewer;

use io::Res;
use itr_core::design::TerrainEdit;
use itr_core::raster::Raster;
use itr_core::scenario::Precision;
use itr_core::Real;
use itr_opt::{Problem, SearchEntry, Sink};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "itr — Inverse Terrain Engine

USAGE:
  itr synth    <dir>                                   write example scenarios (synthetic-valley, diverted-flood, infeasible-problem)
  itr inspect  <dem.tif>                               CRS, extent, nodata, z range, memory estimate
  itr simulate --scenario <s.toml> --out <dir> [--threads N] [--precision f32|f64]
  itr optimize --scenario <s.toml> --out <dir> [--max-sims N] [--max-cell-updates N] [--method cma_es|lq_cma_es|random|sobol|coordinate]
               [--seed N] [--threads N] [--precision f32|f64] [--backend cpu|gpu] [--log-json <events.jsonl>]
  itr optimize --resume <dir> [--threads N]            exact resume (deterministic replay + cache)
  itr serve-eval --scenario <s.toml> [--threads N]     ask/tell JSON-lines evaluator on stdin/stdout (external optimizers)
  itr compare  --baseline <dir> --candidate <dir>
  itr validate [--suite standard] [--precision f32|f64|both]
  itr bench    [--suite mini|standard|large] [--threads N] [--backend cpu|gpu] [--json]
  itr export   --run <dir> --format geotiff|geojson|csv|landxml|dxf [--out <dir>]
  itr view     <run-dir> [--serve [PORT]]

Exit codes: 0 ok, 1 error, 2 infeasible problem (valid result), 3 validation failure.";

#[derive(Default)]
struct Args {
    cmd: String,
    pos: Vec<String>,
    scenario: Option<PathBuf>,
    out: Option<PathBuf>,
    resume: Option<PathBuf>,
    baseline: Option<PathBuf>,
    candidate: Option<PathBuf>,
    run: Option<PathBuf>,
    threads: usize,
    precision: Option<String>,
    max_sims: Option<usize>,
    max_cell_updates: Option<u64>,
    method: Option<String>,
    seed: Option<u64>,
    suite: String,
    json: bool,
    format: String,
    serve: Option<u16>,
    log_json: Option<PathBuf>,
    backend: Option<String>,
}

fn parse() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;
    let mut a = Args { suite: "standard".into(), format: "geotiff".into(), ..Default::default() };
    let mut p = lexopt::Parser::from_env();
    while let Some(arg) = p.next()? {
        match arg {
            Long("scenario") => a.scenario = Some(p.value()?.into()),
            Long("out") => a.out = Some(p.value()?.into()),
            Long("resume") => a.resume = Some(p.value()?.into()),
            Long("baseline") => a.baseline = Some(p.value()?.into()),
            Long("candidate") => a.candidate = Some(p.value()?.into()),
            Long("run") => a.run = Some(p.value()?.into()),
            Long("threads") => a.threads = p.value()?.parse()?,
            Long("precision") => a.precision = Some(p.value()?.string()?),
            Long("max-sims") => a.max_sims = Some(p.value()?.parse()?),
            Long("max-cell-updates") => a.max_cell_updates = Some(p.value()?.parse::<f64>()? as u64),
            Long("method") => a.method = Some(p.value()?.string()?),
            Long("seed") => a.seed = Some(p.value()?.parse()?),
            Long("suite") => a.suite = p.value()?.string()?,
            Long("format") => a.format = p.value()?.string()?,
            Long("json") => a.json = true,
            Long("backend") => a.backend = Some(p.value()?.string()?),
            Long("log-json") => a.log_json = Some(p.value()?.into()),
            Long("serve") => {
                a.serve = Some(match p.optional_value() {
                    Some(v) => v.parse()?,
                    None => 8000,
                })
            }
            Short('h') | Long("help") => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            Value(v) if a.cmd.is_empty() => a.cmd = v.string()?,
            Value(v) => a.pos.push(v.string()?),
            _ => return Err(arg.unexpected()),
        }
    }
    if a.threads == 0 {
        a.threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    }
    Ok(a)
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(1);
        }
    };
    let r = match a.cmd.as_str() {
        "synth" => cmd_synth(&a),
        "inspect" => cmd_inspect(&a),
        "simulate" => with_precision(&a, Cmd::Simulate),
        "optimize" => with_precision(&a, Cmd::Optimize),
        "serve-eval" => with_precision(&a, Cmd::Serve),
        "compare" => cmd_compare(&a),
        "validate" => cmd_validate(&a),
        "bench" => cmd_bench(&a),
        "export" => cmd_export(&a),
        "view" => cmd_view(&a),
        "" => {
            println!("{USAGE}");
            Ok(0)
        }
        c => Err(format!("unknown command `{c}`\n\n{USAGE}")),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn need<'a>(v: &'a Option<PathBuf>, flag: &str) -> Res<&'a PathBuf> {
    v.as_ref().ok_or_else(|| format!("missing --{flag}"))
}

fn mkdir(p: &Path) -> Res<()> {
    std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_json(p: &Path) -> Res<Value> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn cmd_synth(a: &Args) -> Res<i32> {
    let root = PathBuf::from(a.pos.first().map(String::as_str).unwrap_or("examples"));
    synth::all(&root)?;
    eprintln!("wrote examples under {}", root.display());
    Ok(0)
}

fn cmd_inspect(a: &Args) -> Res<i32> {
    let p = PathBuf::from(a.pos.first().ok_or("usage: itr inspect <dem.tif>")?);
    let (r, info) = io::read_geotiff(&p)?;
    let valid: Vec<f64> = r.data.iter().copied().filter(|v| v.is_finite()).collect();
    let (zmin, zmax) = valid.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let n = r.nx * r.ny;
    let v = json!({
        "path": p.display().to_string(),
        "size": [r.nx, r.ny], "cell_size_m": [r.geo.dx, r.geo.dy],
        "extent": {"west": r.geo.origin_x, "north": r.geo.origin_y,
                   "east": r.geo.origin_x + r.nx as f64 * r.geo.dx, "south": r.geo.origin_y - r.ny as f64 * r.geo.dy},
        "epsg": info.epsg, "model_type": match info.model_type { Some(1) => "projected", Some(2) => "geographic (unsupported)", _ => "unknown" },
        "pixel_is_point": info.pixel_is_point, "compression": info.compression,
        "nodata_value": info.nodata, "nodata_cells": n - valid.len(),
        "z_range_m": [zmin, zmax],
        "memory_estimate_mb": {"f32_state_per_workspace": (n * 3 * 4) as f64 / 1e6, "f32_terrain": (n * 4 * 4) as f64 / 1e6,
                                "f64_state_per_workspace": (n * 3 * 8) as f64 / 1e6},
    });
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    Ok(0)
}

#[derive(Clone, Copy)]
enum Cmd {
    Simulate,
    Optimize,
    Serve,
}

fn with_precision(a: &Args, c: Cmd) -> Res<i32> {
    let scen = match (&a.resume, &a.scenario) {
        (Some(d), _) => {
            let st = read_json(&d.join("run.json"))?;
            PathBuf::from(st["scenario"].as_str().ok_or("run.json: no scenario")?)
        }
        (None, Some(s)) => s.clone(),
        _ => return Err("missing --scenario".into()),
    };
    let mut ld = load::load(&scen)?;
    let sc = &mut ld.inputs.scenario;
    if let Some(d) = &a.resume {
        // Resume uses exactly the recorded overrides.
        let st = read_json(&d.join("run.json"))?;
        let o: itr_core::scenario::OptimizerCfg = serde_json::from_value(st["optimizer"].clone()).map_err(|e| e.to_string())?;
        sc.optimizer = o;
        sc.solver.precision = serde_json::from_value(st["precision"].clone()).map_err(|e| e.to_string())?;
        if !st["backend"].is_null() {
            sc.solver.backend = serde_json::from_value(st["backend"].clone()).map_err(|e| e.to_string())?;
        }
    } else {
        if let Some(p) = &a.precision {
            sc.solver.precision = match p.as_str() {
                "f64" => Precision::F64,
                "f32" => Precision::F32,
                _ => return Err("--precision must be f32 or f64".into()),
            };
        }
        if let Some(b) = &a.backend {
            sc.solver.backend = match b.as_str() {
                "cpu" => itr_core::scenario::Backend::Cpu,
                "gpu" => itr_core::scenario::Backend::Gpu,
                _ => return Err("--backend must be cpu or gpu".into()),
            };
        }
        if let Some(n) = a.max_sims {
            sc.optimizer.max_simulations = n;
        }
        if a.max_cell_updates.is_some() {
            sc.optimizer.max_cell_updates = a.max_cell_updates;
        }
        if let Some(m) = &a.method {
            if !["cma_es", "lq_cma_es", "random", "sobol", "coordinate"].contains(&m.as_str()) {
                return Err("--method must be cma_es, lq_cma_es, random, sobol or coordinate".into());
            }
            sc.optimizer.method = m.clone();
        }
        if let Some(s) = a.seed {
            sc.optimizer.seed = s;
        }
    }
    match (sc.solver.precision, c) {
        (Precision::F32, Cmd::Simulate) => simulate::<f32>(a, ld),
        (Precision::F64, Cmd::Simulate) => simulate::<f64>(a, ld),
        (Precision::F32, Cmd::Optimize) => optimize::<f32>(a, ld),
        (Precision::F64, Cmd::Optimize) => optimize::<f64>(a, ld),
        (Precision::F32, Cmd::Serve) => serve::serve(&build::<f32>(&ld, a.threads, true)?, a.threads, ld.inputs.scenario.solver.backend),
        (Precision::F64, Cmd::Serve) => serve::serve(&build::<f64>(&ld, a.threads, true)?, a.threads, ld.inputs.scenario.solver.backend),
    }
}

const CHECKPOINT_BUDGET: usize = 512 << 20;

fn build<T: Real>(ld: &load::Loaded, threads: usize, all_levels: bool) -> Res<Problem<T>> {
    let mut inputs = ld.inputs.clone();
    if !all_levels {
        inputs.scenario.optimizer.fidelity_levels = vec![1];
        inputs.scenario.optimizer.level_budget_share = None;
    }
    let t0 = Instant::now();
    let pb = Problem::<T>::build(inputs, threads, if all_levels { CHECKPOINT_BUDGET } else { 0 })?;
    for l in &pb.levels {
        eprintln!(
            "baseline level {}: {}×{} cells, {} steps, {:.2e} cell-updates, {:.2}s, mass residual {:.1e}",
            l.level,
            l.base.nx(),
            l.base.ny(),
            l.baseline.steps,
            l.baseline.cell_updates as f64,
            l.baseline.wall_s,
            l.baseline.ledger.rel_residual
        );
    }
    eprintln!("problem built in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(pb)
}

fn fine_raster(pb: &Problem<impl Real>, data: Vec<f64>) -> Raster {
    Raster { nx: pb.fine.nx, ny: pb.fine.ny, geo: pb.fine.geo.clone(), data }
}

fn asset_table(r: &itr_core::objective::ObjectiveReport) -> String {
    let mut s = String::new();
    for a in &r.assets {
        s += &format!("  {:<22} peak {:>7.3} m  threshold {:.3} m  exceedance {:.3} m\n", a.name, a.peak_depth_m, a.threshold_m, a.exceedance_m);
    }
    s
}

fn simulate<T: Real>(a: &Args, ld: load::Loaded) -> Res<i32> {
    let out = need(&a.out, "out")?;
    mkdir(out)?;
    let pb = build::<T>(&ld, a.threads, false)?;
    let obj = run::full_objective(&pb);
    let fr = run::full_run(&pb, &obj, &TerrainEdit::default(), None, a.threads)?;
    let s = &fr.summary;
    let zr = pb.fine.z_ref;
    let mut terrain = ld.inputs.dem.clone();
    for (k, v) in terrain.data.iter_mut().enumerate() {
        if pb.fine.wall.data[k] {
            *v = f64::NAN;
        }
    }
    io::write_geotiff(&out.join("terrain.tif"), &terrain, ld.epsg)?;
    io::write_geotiff(&out.join("peak_depth.tif"), &fr.peak, ld.epsg)?;
    io::write_geotiff(&out.join("time_to_peak.tif"), &fr.tpeak, ld.epsg)?;
    run::write_ledger_csv(&out.join("mass_balance.csv"), s)?;
    let frames = run::write_frames(out, "baseline", &fr.frames)?;
    let metrics = json!({
        "kind": "simulation",
        "objective": fr.report,
        "run": {"steps": s.steps, "t_end": s.t_end, "dt_min": s.dt_min, "dt_max": s.dt_max, "dt_mean": s.dt_mean,
                "cell_updates": s.cell_updates, "rows_skipped": s.rows_skipped, "wall_s": s.wall_s,
                "cell_updates_per_s": s.cell_updates as f64 / s.wall_s.max(1e-9), "lts_potential": s.lts_potential,
                "threads": a.threads},
        "mass_balance": {"rel_residual": s.ledger.rel_residual, "abs_residual_m3": s.ledger.abs_residual_m3,
                         "gross_throughput_m3": s.ledger.gross_throughput_m3},
        "z_ref_m": zr,
    });
    io::write_json(&out.join("metrics.json"), &metrics)?;
    io::write_json(&out.join("manifest.json"), &run::manifest(&ld, &pb, json!({"command": "simulate", "threads": a.threads})))?;
    io::write_json(&out.join("viewer_index.json"), &json!({"kind": "simulation", "nx": pb.fine.nx, "ny": pb.fine.ny,
        "geo": pb.fine.geo, "rasters": {"terrain": "terrain.tif", "peak_before": "peak_depth.tif"},
        "assets": viewer_assets(&ld), "baseline": frames}))?;
    eprintln!(
        "simulated {:.0} s in {} steps, {:.2}s wall ({:.2e} cell-updates/s), mass residual {:.2e}\n{}",
        s.t_end,
        s.steps,
        s.wall_s,
        s.cell_updates as f64 / s.wall_s.max(1e-9),
        s.ledger.rel_residual,
        asset_table(&fr.report)
    );
    Ok(0)
}

fn viewer_assets(ld: &load::Loaded) -> Value {
    json!(ld.inputs.assets.iter().map(|(n, m, thr, _)| json!({"name": n, "cells": m.indices(), "threshold_m": thr})).collect::<Vec<_>>())
}

struct FileSink {
    search: io::Jsonl,
    cache: io::Jsonl,
    state: PathBuf,
    t0: Instant,
    best: Option<f64>,
    log: Option<io::Jsonl>,
}

impl Sink for FileSink {
    fn on_eval(&mut self, e: &SearchEntry) {
        self.search.line(e);
        let r = &e.record;
        if r.fitness_feasible && r.status == itr_opt::Status::Ok && self.best.is_none_or(|b| r.j < b) {
            self.best = Some(r.j);
        }
    }
    fn on_fresh(&mut self, r: &itr_opt::EvalRecord) {
        self.cache.line(r);
    }
    fn on_generation(&mut self, st: &Value) {
        let _ = io::write_json(&self.state, st);
        if let Some(l) = &mut self.log {
            l.line(&json!({"event": "generation", "elapsed_s": self.t0.elapsed().as_secs_f64(), "best_feasible_j": self.best, "state": st}));
        }
        eprintln!(
            "[{:>6.1}s] gen {:>3} level {} sims {:>4} cell-updates {:.2e} best feasible J {}",
            self.t0.elapsed().as_secs_f64(),
            st["generation"],
            st["level"],
            st["sims"],
            st["cell_updates"].as_f64().unwrap_or(0.0),
            self.best.map_or("-".into(), |b| format!("{b:.5}"))
        );
    }
}

fn optimize<T: backend::Backends>(a: &Args, ld: load::Loaded) -> Res<i32> {
    let resume = a.resume.is_some();
    let out = if let Some(d) = &a.resume { d.clone() } else { need(&a.out, "out")?.clone() };
    mkdir(&out)?;
    if ld.inputs.scenario.design.is_none() || ld.inputs.primitives.is_empty() {
        return Err("optimize needs a [design] section with primitives".into());
    }
    let sc = &ld.inputs.scenario;
    let scen_abs = std::fs::canonicalize(&ld.scenario_path).map_err(|e| e.to_string())?;
    if !resume {
        io::write_json(&out.join("run.json"), &json!({"scenario": scen_abs.display().to_string(), "optimizer": sc.optimizer,
            "precision": sc.solver.precision, "backend": sc.solver.backend}))?;
    }
    let pb = build::<T>(&ld, a.threads, true)?;
    let mut ev = T::evaluator(&pb, a.threads, sc.solver.backend)?;
    if resume {
        let p = out.join("cache.jsonl");
        if let Ok(s) = std::fs::read_to_string(&p) {
            for line in s.lines() {
                // A torn final line (killed mid-write) is skipped; that evaluation reruns.
                if let Ok(r) = serde_json::from_str::<itr_opt::EvalRecord>(line) {
                    ev.cache.insert(r.key.clone(), r);
                }
            }
        }
        eprintln!("resume: {} cached evaluations", ev.cache.len());
    }
    let mut sink = FileSink {
        search: io::Jsonl::create(&out.join("search.jsonl"), false)?,
        cache: io::Jsonl::create(&out.join("cache.jsonl"), resume)?,
        state: out.join("optimizer_state.json"),
        t0: Instant::now(),
        best: None,
        log: a.log_json.as_deref().map(|p| io::Jsonl::create(p, resume)).transpose()?,
    };
    let t0 = Instant::now();
    let res = itr_opt::search(&pb, &mut ev, &mut sink);
    let search_wall = t0.elapsed().as_secs_f64();
    drop(sink);
    eprintln!("search: {} evaluations, {} simulations, {} cache hits, {:.2e} cell-updates, {:.1}s ({}), utilization U = {:.2}",
        res.evaluations, res.sims, res.cache_hits, res.cell_updates as f64, search_wall, res.stopped_by, res.utilization);
    if ev.batch_sims > 0 {
        eprintln!("gpu backend ran {} of {} simulations (U counts CPU runs only)", ev.batch_sims, res.sims);
    }
    if res.utilization < 0.8 && ev.batch_sims < res.sims {
        let w = a.threads;
        eprintln!("hint: U < 0.8; consider optimizer.population as a multiple of the worker count ({w}), e.g. {} (changes the trace, so never automatic)", w * (sc.optimizer.population.unwrap_or(w) / w).max(1));
    }

    // Final: rerun no-change and best design at full resolution with full outputs.
    let obj = run::full_objective(&pb);
    let base = run::full_run(&pb, &obj, &TerrainEdit::default(), None, a.threads)?;
    let ds = pb.design.as_ref().unwrap();
    let d = &ld.inputs.dem;
    let best = res.best.clone();
    let baseline_j = base.report.j;
    let (edit, cand) = match &best {
        Some(b) if b.record.fitness_feasible => {
            let e = ds.materialize(&b.record.primitives, &d.geo, d.nx, d.ny, &pb.editable);
            let fr = run::full_run(&pb, &obj, &e, Some(&base.summary.monitor_max), a.threads)?;
            (e, Some(fr))
        }
        _ => (TerrainEdit::default(), None),
    };
    let improved = cand.as_ref().is_some_and(|c| c.report.feasible && c.report.j < baseline_j);
    let outcome = match &cand {
        Some(c) if improved && c.report.assets_exceeding == 0 => "improved_meets_thresholds",
        Some(_) if improved => "improved_partial",
        _ => "infeasible",
    };
    // Rasters.
    let zr = pb.fine.z_ref;
    let mut before = d.clone();
    for (k, v) in before.data.iter_mut().enumerate() {
        if pb.fine.wall.data[k] {
            *v = f64::NAN;
        }
    }
    let after = edit.apply(&before);
    let dz = fine_raster(&pb, (0..before.data.len()).map(|k| if before.data[k].is_finite() { edit.get(k % d.nx, k / d.nx) } else { f64::NAN }).collect());
    io::write_geotiff(&out.join("terrain_before.tif"), &before, ld.epsg)?;
    io::write_geotiff(&out.join("terrain_after.tif"), &after, ld.epsg)?;
    io::write_geotiff(&out.join("terrain_delta.tif"), &dz, ld.epsg)?;
    io::write_geotiff(&out.join("peak_depth_before.tif"), &base.peak, ld.epsg)?;
    io::write_geotiff(&out.join("time_to_peak_before.tif"), &base.tpeak, ld.epsg)?;
    run::write_ledger_csv(&out.join("mass_balance_before.csv"), &base.summary)?;
    let fb = run::write_frames(&out, "before", &base.frames)?;
    let mut viewer = json!({"kind": "optimization", "nx": pb.fine.nx, "ny": pb.fine.ny, "geo": pb.fine.geo,
        "assets": viewer_assets(&ld), "before": fb,
        "rasters": {"terrain_before": "terrain_before.tif", "terrain_after": "terrain_after.tif", "terrain_delta": "terrain_delta.tif",
                    "peak_before": "peak_depth_before.tif"}});
    let mut metrics = json!({
        "kind": "optimization",
        "outcome": outcome,
        "method": res.method, "levels": res.levels, "stopped_by": res.stopped_by,
        "budget": {"simulations": res.sims, "cell_updates": res.cell_updates, "evaluations": res.evaluations,
                   "cache_hits": res.cache_hits, "search_wall_s": search_wall, "utilization": res.utilization,
                   "backend": sc.solver.backend, "batch_backend_simulations": ev.batch_sims},
        "objective_normalization": format!("J = J_risk + λ·J_earth with λ = {} per cost unit; J_risk in metres (softplus, τ = {} m)",
            obj.def.earth_weight, obj.def.tau_m),
        "baseline": base.report,
        "z_ref_m": zr,
        "pareto_front": res.pareto.iter().map(|e| json!({"j": e.record.j, "j_risk": e.record.report.as_ref().map(|r| r.j_risk),
            "earthwork_m3": e.record.report.as_ref().map(|r| r.cut_m3 + r.fill_m3), "eval_index": e.eval_index})).collect::<Vec<_>>(),
    });
    if let Some(c) = &cand {
        let (dd, dsum) = run::delta(&base.peak, &c.peak, pb.fine.geo.cell_area());
        io::write_geotiff(&out.join("peak_depth_after.tif"), &c.peak, ld.epsg)?;
        io::write_geotiff(&out.join("time_to_peak_after.tif"), &c.tpeak, ld.epsg)?;
        io::write_geotiff(&out.join("delta_peak_depth.tif"), &dd, ld.epsg)?;
        run::write_ledger_csv(&out.join("mass_balance.csv"), &c.summary)?;
        viewer["after"] = run::write_frames(&out, "after", &c.frames)?;
        viewer["rasters"]["peak_after"] = json!("peak_depth_after.tif");
        viewer["rasters"]["delta_peak"] = json!("delta_peak_depth.tif");
        let b = best.as_ref().unwrap();
        // The full-output rerun must reproduce the search evaluation bitwise (§6.5, §9.1.1).
        // With a robust objective the full-output run is the nominal member.
        let search_j = b.record.members.first().map_or(b.record.j, |m| m.j);
        // On the GPU backend the CPU rerun agrees within the D3 tolerance only (§7.8).
        let gpu = sc.solver.backend == itr_core::scenario::Backend::Gpu;
        let rel = (c.report.j - search_j).abs() / search_j.abs().max(1e-12);
        let reproduced = if gpu { rel <= 1e-4 } else { c.report.j.to_bits() == search_j.to_bits() };
        if gpu {
            metrics["reproduction"] = json!({"mode": "gpu search, cpu f32 rerun (D3)", "relative_j_difference": rel, "tolerance": 1e-4});
        }
        if let Some(r) = &sc.robustness {
            metrics["robust"] = json!({"objective": "E[J] + beta * CVaR_alpha(J), equal member weights", "beta": r.beta, "alpha": r.alpha,
                "j_robust": b.record.j, "members": b.record.members, "uncertainty_model": r.uncertainty_model});
        }
        metrics["candidate"] = serde_json::to_value(&c.report).unwrap();
        metrics["peak_depth_change"] = dsum;
        metrics["reproduced_bitwise"] = json!(!gpu && reproduced);
        metrics["candidate_run"] = json!({"steps": c.summary.steps, "mass_rel_residual": c.summary.ledger.rel_residual});
        metrics["unresolved_primitives"] = json!(b.record.unresolved);
        // Ablation / Shapley attribution at the finest search level (§19.2).
        let li = pb.levels.len() - 1;
        let mut ev = itr_opt::Evaluator::new(&pb, a.threads);
        let ab = itr_opt::ablation::ablate(&pb, &mut ev, li, &b.record.primitives);
        for at in &ab.attributions {
            eprintln!("ablation: primitive {} ({:?}) Shapley ΔJ {} leave-one-out +{:.5}{}", at.index, at.kind,
                at.shapley_j_reduction.map_or("n/a".into(), |v| format!("{v:.5}")), at.leave_one_out_j_increase,
                if at.redundant { " (redundant)" } else { "" });
        }
        metrics["ablation"] = json!({"file": "ablation.json", "exact": ab.exact, "simulations": ab.simulations,
            "redundant_primitives": ab.attributions.iter().filter(|x| x.redundant).map(|x| x.index).collect::<Vec<_>>(),
            "shapley_j_reduction": ab.attributions.iter().map(|x| json!({"index": x.index, "value": x.shapley_j_reduction})).collect::<Vec<_>>()});
        io::write_json(&out.join("ablation.json"), &serde_json::to_value(&ab).unwrap())?;
        let feats: Vec<(Vec<(f64, f64)>, Value)> = b
            .record
            .primitives
            .iter()
            .enumerate()
            .filter(|(_, p)| p.height != 0.0)
            .map(|(i, p)| {
                let (x0, y0, x1, y1) = p.bbox();
                let ring = footprint(p).unwrap_or(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]);
                (ring, json!({"index": i, "kind": p.kind, "cx": p.cx, "cy": p.cy, "length_m": p.length, "width_m": p.width,
                    "angle_deg": p.angle_deg, "height_m": p.height, "design_space": b.space,
                    "theta_normalized": slice_theta(&ds.specs, &b.u, i, b.space == "corridor")}))
            })
            .collect();
        io::write_json(&out.join("earthworks.geojson"), &io::polys_to_geojson(&feats))?;
        if !reproduced {
            eprintln!("warning: full-output rerun J {} differs from search J {}", c.report.j, search_j);
        }
    }
    io::write_json(&out.join("metrics.json"), &metrics)?;
    io::write_json(&out.join("viewer_index.json"), &viewer)?;
    io::write_json(&out.join("manifest.json"), &run::manifest(&ld, &pb, json!({"command": "optimize", "threads": a.threads,
        "seed": sc.optimizer.seed, "method": sc.optimizer.method, "resumed": resume})))?;
    eprintln!("baseline J {:.5}\n{}", baseline_j, asset_table(&base.report));
    match &cand {
        Some(c) => eprintln!(
            "best J {:.5} (cut {:.0} m³, fill {:.0} m³, guard worsening {:.3} m) → {outcome}\n{}",
            c.report.j,
            c.report.cut_m3,
            c.report.fill_m3,
            c.report.guard_max_worsening_m,
            asset_table(&c.report)
        ),
        None => eprintln!("no feasible candidate → {outcome}"),
    }
    Ok(if outcome == "infeasible" { 2 } else { 0 })
}

fn slice_theta(specs: &[itr_core::design::PrimitiveSpec], u: &[f64], prim: usize, corridor: bool) -> Vec<f64> {
    let mut k = 0;
    for (i, s) in specs.iter().enumerate() {
        let mound = matches!(s.kind, itr_core::design::PrimitiveKind::Mound);
        let n = match (corridor, mound) {
            (false, true) => 4,
            (false, false) => 6,
            (true, true) => 3,
            (true, false) => 4,
        };
        if i == prim {
            return u.get(k..k + n).map(<[f64]>::to_vec).unwrap_or_default();
        }
        k += n;
    }
    vec![]
}

/// Outline of a primitive's support (stadium for linear kinds, circle for mounds).
fn footprint(p: &itr_core::design::Primitive) -> Option<Vec<(f64, f64)>> {
    let ((x0, y0), (x1, y1)) = p.segment();
    let r = 0.5 * p.width;
    let ang = (y1 - y0).atan2(x1 - x0);
    let mut ring = vec![];
    for k in 0..=16 {
        let t = ang - std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 16.0;
        ring.push((x1 + r * t.cos(), y1 + r * t.sin()));
    }
    for k in 0..=16 {
        let t = ang + std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 16.0;
        ring.push((x0 + r * t.cos(), y0 + r * t.sin()));
    }
    ring.push(ring[0]);
    Some(ring)
}

fn cmd_compare(a: &Args) -> Res<i32> {
    let b = need(&a.baseline, "baseline")?;
    let c = need(&a.candidate, "candidate")?;
    let mb = read_json(&b.join("metrics.json"))?;
    let mc = read_json(&c.join("metrics.json"))?;
    let base_obj = if mb["kind"] == "simulation" { mb["objective"].clone() } else { mb["baseline"].clone() };
    let cand_obj = if mc["kind"] == "optimization" { mc["candidate"].clone() } else { mc["objective"].clone() };
    let pick = |dir: &Path, names: &[&str]| names.iter().map(|n| dir.join(n)).find(|p| p.exists());
    let pb = pick(b, &["peak_depth.tif", "peak_depth_before.tif"]).ok_or("baseline has no peak depth raster")?;
    let pc = pick(c, &["peak_depth_after.tif", "peak_depth.tif"]).ok_or("candidate has no peak depth raster")?;
    let (rb, _) = io::read_geotiff(&pb)?;
    let (rc, _) = io::read_geotiff(&pc)?;
    if rb.nx != rc.nx || rb.ny != rc.ny {
        return Err("runs are on different grids".into());
    }
    let (_, dsum) = run::delta(&rb, &rc, rb.geo.cell_area());
    let mut assets = vec![];
    if let (Some(x), Some(y)) = (base_obj["assets"].as_array(), cand_obj["assets"].as_array()) {
        for (p, q) in x.iter().zip(y) {
            assets.push(json!({"name": p["name"], "peak_before_m": p["peak_depth_m"], "peak_after_m": q["peak_depth_m"],
                "change_m": q["peak_depth_m"].as_f64().unwrap_or(0.0) - p["peak_depth_m"].as_f64().unwrap_or(0.0)}));
        }
    }
    let v = json!({"baseline": b.display().to_string(), "candidate": c.display().to_string(),
        "j_before": base_obj["j"], "j_after": cand_obj["j"], "assets": assets, "peak_depth_change": dsum,
        "cut_m3": cand_obj["cut_m3"], "fill_m3": cand_obj["fill_m3"],
        "guard_max_worsening_m": cand_obj["guard_max_worsening_m"], "feasible": cand_obj["feasible"]});
    println!("{}", serde_json::to_string_pretty(&v).unwrap());
    Ok(0)
}

fn cmd_validate(a: &Args) -> Res<i32> {
    if a.suite != "standard" {
        return Err("only --suite standard is available".into());
    }
    let prec = a.precision.as_deref().unwrap_or("both");
    let mut reps = vec![];
    if prec == "f64" || prec == "both" {
        reps.extend(itr_hydro::validation::suite::<f64>());
    }
    if prec == "f32" || prec == "both" {
        reps.extend(itr_hydro::validation::suite::<f32>());
    }
    let mut fail = 0;
    for r in &reps {
        eprintln!(
            "{} {:<22} {:<4} {:<34} {:>10.3e} (≤ {:.1e}) {}",
            if r.passed { "PASS" } else { "FAIL" },
            r.name,
            r.precision,
            r.metric,
            r.value,
            r.threshold,
            r.notes
        );
        fail += (!r.passed) as usize;
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&reps).unwrap());
    }
    eprintln!("{} cases, {} failed", reps.len(), fail);
    Ok(if fail > 0 { 3 } else { 0 })
}

fn bench_case<T: Real>(n: usize, threads: usize) -> Value {
    use itr_core::model::MonitorSet;
    use itr_hydro::validation::{params, terrain};
    use itr_hydro::{PreparedTerrain, Solver, Workspace};
    let dx = 2.0;
    let l = n as f64 * dx;
    let base = terrain::<T>(n, n, dx, |x, y| 10.0 + 0.002 * (l - x) + 0.4 * (x / 23.0).sin() * (y / 17.0).cos(), |_, _| false, 0.035, &[]);
    let mut p = params(600.0, 60.0);
    p.initial_depth = 0.05;
    p.rain = itr_core::scenario::Hyetograph::from_mm_h(&[[0.0, 50.0]]);
    let terr = PreparedTerrain::new(base.clone());
    let mut ws = Workspace::<T>::new(base.layout, &MonitorSet::default(), threads, 16);
    let s = Solver::<T>::new(p).run_impl(&mut ws, &terr, None, &mut itr_core::model::NoopObserver).unwrap();
    let rate = s.cell_updates as f64 / s.wall_s;
    // Bytes/cell-update model: 3 state arrays read+written (in place) + ~4 terrain arrays read.
    let bpu = (3 * 2 + 4) * core::mem::size_of::<T>();
    json!({"cells": n * n, "nx": n, "ny": n, "precision": T::NAME, "threads": threads, "steps": s.steps,
        "simulated_s": s.t_end, "dt_min": s.dt_min, "dt_mean": s.dt_mean, "dt_max": s.dt_max, "wall_s": s.wall_s,
        "cell_updates": s.cell_updates, "cell_updates_per_s": rate, "bytes_per_cell_update_model": bpu,
        "achieved_gb_s_model": rate * bpu as f64 / 1e9, "mass_rel_residual": s.ledger.rel_residual})
}

fn cmd_bench(a: &Args) -> Res<i32> {
    match a.backend.as_deref() {
        Some("gpu") => return gpu_bench(a),
        None | Some("cpu") => {}
        _ => return Err("--backend must be cpu or gpu".into()),
    }
    let sizes: &[usize] = match a.suite.as_str() {
        "standard" => &[128, 512],
        "large" => &[128, 512, 1024, 2048],
        "mini" => &[128],
        _ => return Err("--suite must be mini, standard or large".into()),
    };
    let mut cases = vec![];
    for &n in sizes {
        for prec in ["f32", "f64"] {
            for th in [1, a.threads] {
                if th == a.threads && th == 1 && cases.iter().any(|c: &Value| c["nx"] == n && c["precision"] == prec) {
                    continue;
                }
                let v = if prec == "f32" { bench_case::<f32>(n, th) } else { bench_case::<f64>(n, th) };
                eprintln!("{n:>5}² {prec} threads {th:>2}: {:.3e} cell-updates/s ({} steps, {:.2}s)", v["cell_updates_per_s"].as_f64().unwrap(), v["steps"], v["wall_s"].as_f64().unwrap());
                cases.push(v);
            }
        }
    }
    let host = json!({"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "backend": "cpu", "engine": env!("CARGO_PKG_VERSION")});
    let v = json!({"host": host, "suite": a.suite, "cases": cases});
    if a.json {
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
    }
    Ok(0)
}

#[cfg(not(feature = "gpu"))]
fn gpu_bench(_: &Args) -> Res<i32> {
    Err("this build has no GPU backend; rebuild with `cargo build --release -p itr-cli --features gpu`".into())
}

/// GPU throughput (spec §7.8): one large domain and batches of small candidate domains
/// (population-parallel), with the CPU f32 single-thread run of the same case for
/// reference and the max |Δ peak depth| between the two.
#[cfg(feature = "gpu")]
fn gpu_bench(a: &Args) -> Res<i32> {
    use itr_core::model::{MonitorSet, NoopObserver};
    use itr_hydro::validation::{params, terrain};
    use itr_hydro::{PreparedTerrain, Solver, Workspace};
    let mut g = itr_gpu::GpuSolver::new()?;
    let adapter = g.adapter_info();
    eprintln!("adapter: {adapter}");
    let cases: &[(usize, usize)] = match a.suite.as_str() {
        "mini" => &[(128, 1), (128, 16)],
        "standard" => &[(128, 1), (128, 16), (128, 64), (512, 1)],
        "large" => &[(128, 64), (512, 1), (512, 8), (1024, 1), (2048, 1)],
        _ => return Err("--suite must be mini, standard or large".into()),
    };
    let mut out = vec![];
    for &(n, batch) in cases {
        let dx = 2.0;
        let l = n as f64 * dx;
        let base = terrain::<f32>(n, n, dx, |x, y| 10.0 + 0.002 * (l - x) + 0.4 * (x / 23.0).sin() * (y / 17.0).cos(), |_, _| false, 0.035, &[]);
        let mut p = params(600.0, 60.0);
        p.initial_depth = 0.05;
        p.rain = itr_core::scenario::Hyetograph::from_mm_h(&[[0.0, 50.0]]);
        let mon = MonitorSet::new((0..(n * n) as u32).step_by(97).collect());
        let ts: Vec<PreparedTerrain<f32>> = (0..batch).map(|_| PreparedTerrain::new(base.clone())).collect();
        let refs: Vec<&PreparedTerrain<f32>> = ts.iter().collect();
        g.run_batch(&p, &refs[..1], &mon)?; // warm-up (pipeline + buffers)
        let t0 = std::time::Instant::now();
        let r = g.run_batch(&p, &refs, &mon)?;
        let wall = t0.elapsed().as_secs_f64();
        let upd: u64 = r.iter().map(|x| x.steps * (n * n) as u64).sum();
        let mut ws = Workspace::<f32>::new(base.layout, &mon, 1, 32);
        let c = Solver::<f32>::new(p).run_impl(&mut ws, &ts[0], None, &mut NoopObserver).map_err(|e| format!("{e:?}"))?;
        let dmax = r[0].monitor_max.iter().zip(&c.monitor_max).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let rate = upd as f64 / wall;
        let cpu = c.cell_updates as f64 / c.wall_s;
        eprintln!("{n:>5}² x{batch:<3} gpu {rate:.3e} cell-updates/s ({} steps, {wall:.2}s); cpu f32 1 thread {cpu:.3e}; max |Δ peak| {dmax:.2e} m", r[0].steps);
        out.push(json!({"nx": n, "ny": n, "batch": batch, "precision": "f32", "steps": r[0].steps, "wall_s": wall,
            "cell_updates_per_s": rate, "cpu_f32_1thread_cell_updates_per_s": cpu, "cpu_steps": c.steps,
            "max_abs_peak_depth_diff_m": dmax, "bad": r.iter().any(|x| x.bad)}));
    }
    let v = json!({"host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "gpu", "adapter": adapter,
        "engine": env!("CARGO_PKG_VERSION")}, "suite": a.suite, "cases": out});
    if a.json {
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
    }
    Ok(0)
}

fn cmd_export(a: &Args) -> Res<i32> {
    let run = need(&a.run, "run")?;
    let out = a.out.clone().unwrap_or_else(|| run.join("export"));
    mkdir(&out)?;
    match a.format.as_str() {
        "geotiff" => {
            let mut n = 0;
            for e in std::fs::read_dir(run).map_err(|e| e.to_string())? {
                let p = e.map_err(|e| e.to_string())?.path();
                if p.extension().is_some_and(|x| x == "tif") {
                    // Re-encode (validates the file and normalizes to Float32 Deflate).
                    let (r, info) = io::read_geotiff(&p)?;
                    io::write_geotiff(&out.join(p.file_name().unwrap()), &r, info.epsg)?;
                    n += 1;
                }
            }
            eprintln!("exported {n} GeoTIFFs to {}", out.display());
        }
        "geojson" => {
            let p = run.join("earthworks.geojson");
            std::fs::copy(&p, out.join("earthworks.geojson")).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        "csv" => {
            let s = std::fs::read_to_string(run.join("search.jsonl")).map_err(|e| e.to_string())?;
            let mut csv = String::from("eval_index,level,generation,status,feasible,j,violation,cut_m3,fill_m3,steps,warm_start_steps,cell_updates,cached\n");
            for line in s.lines() {
                let e: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
                let r = &e["record"];
                csv += &format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    e["eval_index"], e["level"], e["generation"], r["status"].as_str().unwrap_or(""), r["fitness_feasible"], r["j"], r["violation"],
                    r["report"]["cut_m3"], r["report"]["fill_m3"], r["steps"], r["warm_start_steps"], r["cell_updates"], e["cached"]
                );
            }
            std::fs::write(out.join("search.csv"), csv).map_err(|e| e.to_string())?;
        }
        "landxml" | "dxf" => {
            let rd = |n: &str| io::read_geotiff(&run.join(n)).map(|x| x.0);
            let (before, after, delta) = (rd("terrain_before.tif")?, rd("terrain_after.tif")?, rd("terrain_delta.tif")?);
            let ew = read_json(&run.join("earthworks.geojson")).unwrap_or(json!({"type": "FeatureCollection", "features": []}));
            if a.format == "landxml" {
                let epsg = io::read_geotiff(&run.join("terrain_after.tif")).ok().and_then(|x| x.1.epsg).filter(|e| *e != 32767);
                std::fs::write(out.join("design.xml"), cad::landxml(&before, &after, &delta, &ew, epsg, 3)).map_err(|e| e.to_string())?;
                eprintln!("wrote {}", out.join("design.xml").display());
            } else {
                std::fs::write(out.join("design.dxf"), cad::dxf(&after, &delta, &ew)).map_err(|e| e.to_string())?;
                eprintln!("wrote {}", out.join("design.dxf").display());
            }
        }
        f => return Err(format!("unknown --format {f} (geotiff, geojson, csv, landxml, dxf)")),
    }
    Ok(0)
}

fn cmd_view(a: &Args) -> Res<i32> {
    let dir = PathBuf::from(a.pos.first().ok_or("usage: itr view <run-dir> [--serve [PORT]]")?);
    if !dir.join("viewer_index.json").exists() {
        return Err(format!("{}: no viewer_index.json (run simulate/optimize first)", dir.display()));
    }
    viewer::write_bundle(&dir)?;
    eprintln!("viewer written: {}", dir.join("index.html").display());
    if let Some(port) = a.serve {
        viewer::serve(&dir, port)?;
    }
    Ok(0)
}
