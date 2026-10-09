//! `itr serve-eval` (spec §19.2): newline-delimited JSON ask/tell evaluator over
//! stdin/stdout, so external optimizers (BoTorch, Optuna, Nevergrad, pymoo, …) can drive
//! the engine without any of them becoming a dependency. The cache, exact early aborts
//! and fidelity levels stay active. Log lines go to stderr; stdout carries only protocol.
//!
//! Requests (one JSON object per line):
//! - `{"id": 1, "cmd": "info"}` → dimension, parameter names, levels, baseline J.
//! - `{"id": 2, "cmd": "eval", "x": [[…], …], "level": 1, "incumbent": null}`
//!   `x` rows are points in the unit box [0,1]^n (values outside are clamped); `level`
//!   is a fidelity coarsening factor from the scenario (default: finest); `incumbent`
//!   enables exact incumbent abort (elitist callers only).
//! - `{"id": 3, "cmd": "quit"}`.
//!
//! Each `eval` reply has one result per row: status, feasible, j (objective), violation
//! (Deb), per-asset peaks, cut/fill, simulated cell-updates, and the cache key.

use crate::io::Res;
use crate::backend::Backends;
use itr_core::scenario::Backend;
use itr_opt::{Evaluator, Problem};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

pub fn serve<T: Backends>(pb: &Problem<T>, workers: usize, backend: Backend) -> Res<i32> {
    let ds = pb.design.as_ref().ok_or("serve-eval needs a [design] section with primitives")?;
    let mut ev = T::evaluator(pb, workers, backend)?;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let send = |out: &mut std::io::StdoutLock<'_>, v: Value| {
        let _ = writeln!(out, "{v}");
        let _ = out.flush();
    };
    send(&mut out, json!({"event": "ready", "protocol": "itr-ask-tell/1", "dimension": ds.dimension()}));
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                send(&mut out, json!({"error": format!("bad JSON: {e}")}));
                continue;
            }
        };
        let id = req["id"].clone();
        match req["cmd"].as_str() {
            Some("info") => {
                let lv = pb.levels.last().unwrap();
                let base = lv.objective.evaluate(&lv.baseline_max, &Default::default(), Some(&lv.baseline_max));
                send(&mut out, json!({"id": id, "dimension": ds.dimension(), "param_names": ds.param_names(),
                    "levels": pb.levels.iter().map(|l| l.level).collect::<Vec<_>>(), "baseline_j": base.j,
                    "baseline_assets": base.assets, "problem_hash": pb.problem_hash,
                    "objective": "minimize j subject to feasible (violation == 0); ties broken by Deb's rules"}));
            }
            Some("eval") => {
                let level = req["level"].as_u64().map(|v| v as usize).unwrap_or(1);
                let Some(li) = pb.levels.iter().position(|l| l.level == level) else {
                    send(&mut out, json!({"id": id, "error": format!("level {level} is not in fidelity_levels")}));
                    continue;
                };
                let Some(rows) = req["x"].as_array() else {
                    send(&mut out, json!({"id": id, "error": "x must be an array of points"}));
                    continue;
                };
                let mut cands = vec![];
                for r in rows {
                    let u: Vec<f64> = r.as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.5).clamp(0.0, 1.0)).collect()).unwrap_or_default();
                    if u.len() != ds.dimension() {
                        cands.clear();
                        break;
                    }
                    cands.push(Evaluator::prepare(pb, li, &u));
                }
                if cands.len() != rows.len() {
                    send(&mut out, json!({"id": id, "error": format!("every point must have {} coordinates", ds.dimension())}));
                    continue;
                }
                let inc = req["incumbent"].as_f64();
                let recs = ev.eval_batch(pb, li, &cands, inc);
                let res: Vec<Value> = recs
                    .iter()
                    .map(|r| {
                        json!({"status": r.status, "feasible": r.fitness_feasible, "j": r.j, "violation": r.violation,
                            "breach_sync": r.breach_sync, "assets": r.report.as_ref().map(|x| &x.assets),
                            "cut_m3": r.report.as_ref().map(|x| x.cut_m3), "fill_m3": r.report.as_ref().map(|x| x.fill_m3),
                            "guard_max_worsening_m": r.report.as_ref().map(|x| x.guard_max_worsening_m),
                            "static_violations": r.static_violations, "cell_updates": r.cell_updates, "key": r.key,
                            "unresolved": r.unresolved})
                    })
                    .collect();
                send(&mut out, json!({"id": id, "results": res, "simulations_total": ev.sims, "cache_hits_total": ev.cache_hits}));
            }
            Some("quit") => {
                send(&mut out, json!({"id": id, "event": "bye"}));
                break;
            }
            other => send(&mut out, json!({"id": id, "error": format!("unknown cmd {other:?} (info, eval, quit)")})),
        }
    }
    Ok(0)
}
