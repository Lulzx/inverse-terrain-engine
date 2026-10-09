//! Search driver (spec §6.4–6.6): multi-fidelity schedule, budget accounting, archive,
//! Pareto front, streaming sink, and exact resume by deterministic replay through the
//! evaluation cache.

use crate::cmaes::CmaEs;
use crate::evaluator::{EvalRecord, Evaluator, Prepared, Status};
use crate::problem::Problem;
use crate::search::{CompassSearch, Fitness, Optimizer, RandomSearch};
use itr_core::Real;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Streaming output of the search (JSONL files in the CLI).
pub trait Sink {
    fn on_eval(&mut self, _rec: &SearchEntry) {}
    fn on_fresh(&mut self, _rec: &EvalRecord) {}
    fn on_generation(&mut self, _state: &serde_json::Value) {}
}
pub struct NullSink;
impl Sink for NullSink {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchEntry {
    pub eval_index: usize,
    pub level: usize,
    pub generation: usize,
    pub u: Vec<f64>,
    pub cached: bool,
    pub record: EvalRecord,
    /// Design space of `u`: "free" or "corridor" (§7.11.4).
    #[serde(default = "free_space")]
    pub space: String,
}

fn free_space() -> String {
    "free".into()
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub method: String,
    pub best: Option<SearchEntry>,
    /// Feasible full-resolution evaluations non-dominated in (J_risk, earthwork cost).
    pub pareto: Vec<SearchEntry>,
    pub sims: usize,
    pub cache_hits: usize,
    pub cell_updates: u64,
    pub evaluations: usize,
    pub levels: Vec<usize>,
    pub stopped_by: String,
    pub final_state: serde_json::Value,
    /// Achieved candidate-parallel utilization U (§7.11.5).
    pub utilization: f64,
}

pub struct SearchCfg {
    pub workers: usize,
}

fn make_opt<T: Real>(
    pb: &Problem<T>,
    method: &str,
    n: usize,
    mean: Vec<f64>,
    sigma: f64,
    seed: u64,
    corridor: Option<&crate::corridor::CorridorSpace>,
) -> Box<dyn Optimizer> {
    let pop = pb.inputs.scenario.optimizer.population;
    let restart = corridor.map(|c| c.restart_means()).unwrap_or_default();
    match method {
        "random" => Box::new(RandomSearch::new(n, pop.unwrap_or(8), seed)),
        "sobol" => Box::new(crate::search::SobolSearch::new(n, pop.unwrap_or(8), seed)),
        "coordinate" => Box::new(CompassSearch::new(mean, sigma.max(0.05))),
        "lq_cma_es" => {
            let mut c = CmaEs::new(n, pop, mean, sigma, seed);
            c.lq = Some(crate::cmaes::Lq::default());
            c.restart_means = restart;
            Box::new(c)
        }
        _ => {
            let mut c = CmaEs::new(n, pop, mean, sigma, seed);
            c.restart_means = restart;
            Box::new(c)
        }
    }
}

fn dominates(a: &EvalRecord, b: &EvalRecord) -> bool {
    let (ra, rb) = (a.report.as_ref().unwrap(), b.report.as_ref().unwrap());
    let (a1, a2, b1, b2) = (ra.j_risk, ra.cut_m3 + ra.fill_m3, rb.j_risk, rb.cut_m3 + rb.fill_m3);
    a1 <= b1 && a2 <= b2 && (a1 < b1 || a2 < b2)
}

pub fn pareto(entries: &[SearchEntry]) -> Vec<SearchEntry> {
    let feas: Vec<&SearchEntry> = entries.iter().filter(|e| e.record.status == Status::Ok && e.record.fitness_feasible).collect();
    let mut out: Vec<SearchEntry> = vec![];
    for e in &feas {
        if !feas.iter().any(|o| dominates(&o.record, &e.record)) && !out.iter().any(|o| o.record.key == e.record.key) {
            out.push((*e).clone());
        }
    }
    out.sort_by(|a, b| a.record.j.total_cmp(&b.record.j));
    out
}

/// Budget accounting by first occurrence of a key *in this search*, so a resumed run
/// (whose earlier evaluations come back from the cache) spends budget exactly like an
/// uninterrupted one and the trace is identical.
#[derive(Default)]
struct Accounting {
    seen: std::collections::HashSet<String>,
    sims: usize,
    cell_updates: u64,
    hits: usize,
}

impl Accounting {
    fn run<T: Real>(
        &mut self,
        ev: &mut Evaluator<T>,
        pb: &Problem<T>,
        li: usize,
        cands: &[Prepared],
        inc: Option<f64>,
        sink: &mut dyn Sink,
    ) -> Vec<(EvalRecord, bool)> {
        let known: Vec<bool> = cands.iter().map(|c| ev.cache.contains_key(&Evaluator::<T>::cache_key(c, inc))).collect();
        let recs = ev.eval_batch(pb, li, cands, inc);
        let mut out = vec![];
        for (r, known) in recs.into_iter().zip(known) {
            let first = self.seen.insert(r.key.clone());
            if first {
                if r.status != Status::StaticInfeasible {
                    self.sims += 1;
                }
                self.cell_updates += r.cell_updates;
                if !known {
                    sink.on_fresh(&r);
                }
            } else {
                self.hits += 1;
            }
            out.push((r, !first));
        }
        out
    }
}

/// Run the configured search. `ev.cache` may be pre-filled (resume): replay is exact
/// because every proposal is a deterministic function of the seed and prior results.
pub fn search<T: Real>(pb: &Problem<T>, ev: &mut Evaluator<T>, sink: &mut dyn Sink) -> SearchResult {
    let oc = pb.inputs.scenario.optimizer.clone();
    let ds = pb.design.as_ref().expect("optimize requires [design] with primitives");
    // Corridor-native search space when available; free placements are still sampled.
    let corridor = pb.corridor.as_ref();
    let in_corr = corridor.is_some();
    let n = corridor.map_or(ds.dimension(), |c| c.dimension());
    let space = if in_corr { "corridor" } else { "free" }.to_string();
    let free_frac = pb.inputs.scenario.design.as_ref().map_or(0.0, |d| d.free_placement_fraction);
    let mut free_rng = crate::rng::Rand::new(oc.seed ^ 0xF4EE);
    let nl = pb.levels.len();
    let shares: Vec<f64> = match &oc.level_budget_share {
        Some(s) => s.clone(),
        None => vec![1.0 / nl as f64; nl],
    };
    let tot: f64 = shares.iter().sum();
    // Incumbent abort bounds the nominal J only; not valid for the robust objective.
    let elitist = !matches!(oc.method.as_str(), "cma_es" | "lq_cma_es") && pb.inputs.scenario.robustness.is_none();
    let mut all: Vec<SearchEntry> = vec![];
    let mut eval_index = 0;
    let mut generation = 0;
    let mut stopped_by = "max_simulations".to_string();
    let mut mean = corridor.map_or(vec![0.5; n], |c| c.heuristic_mean(0.5));
    let mut sigma = oc.initial_sigma;
    let mut seeds: Vec<Vec<f64>> = vec![];
    let mut final_state = serde_json::Value::Null;
    let mut used_before = 0usize;
    let mut acct = Accounting::default();
    'levels: for li in 0..nl {
        let level_budget = ((oc.max_simulations as f64) * shares[..=li].iter().sum::<f64>() / tot).round() as usize;
        let mut opt = make_opt(pb, &oc.method, n, mean.clone(), sigma, oc.seed.wrapping_add(li as u64 * 0x9E37_79B9), corridor);
        let mut level_entries: Vec<SearchEntry> = vec![];
        // Seed: re-evaluate the previous level's top candidates at this level.
        if !seeds.is_empty() {
            let cands: Vec<Prepared> = seeds.iter().map(|u| Evaluator::prepare_in(pb, li, u, in_corr)).collect();
            let recs = acct.run(ev, pb, li, &cands, None, sink);
            for (u, (r, cached)) in seeds.iter().zip(recs) {
                let e = SearchEntry { eval_index, level: pb.levels[li].level, generation, u: u.clone(), cached, record: r, space: space.clone() };
                eval_index += 1;
                sink.on_eval(&e);
                level_entries.push(e);
            }
        }
        loop {
            if acct.sims >= level_budget {
                break;
            }
            if let Some(m) = oc.max_cell_updates
                && acct.cell_updates >= m {
                    stopped_by = "max_cell_updates".into();
                    break 'levels;
                }
            let bs = opt.batch_size();
            let mut us: Vec<Vec<f64>> = vec![];
            let mut cands: Vec<Prepared> = vec![];
            for slot in 0..bs {
                let mut attempt = 0;
                let mut best: Option<(Vec<f64>, Prepared)> = None;
                while let Some(u) = opt.propose(slot, attempt) {
                    let c = Evaluator::prepare_in(pb, li, &u, in_corr);
                    let ok = c.violations.is_empty();
                    best = Some((u, c));
                    attempt += 1;
                    if ok || attempt >= oc.max_static_attempts {
                        break;
                    }
                }
                let (u, c) = best.expect("optimizer produced no proposal");
                us.push(u);
                cands.push(c);
            }
            let sims_before = acct.sims;
            let inc = if elitist { opt.incumbent() } else { None };
            // Surrogate pre-screening (lq-CMA-ES): only selected proposals are simulated.
            let mask = opt.select(&us);
            let sel: Vec<usize> = (0..us.len()).filter(|&k| mask[k]).collect();
            let mut sel_cands: Vec<Prepared> = sel.iter().map(|&k| std::mem::replace(&mut cands[k], Prepared::empty())).collect();
            // Escape hatch (§7.11.4): in corridor mode, a share of each generation samples
            // free placements. They enter the archive (and can win) but are not told to
            // the corridor-space optimizer.
            let n_free = if in_corr { (free_frac * bs as f64).ceil() as usize } else { 0 };
            let free_u = crate::corridor::free_proposals(ds.dimension(), n_free, &mut free_rng);
            sel_cands.extend(free_u.iter().map(|u| Evaluator::prepare_in(pb, li, u, false)));
            let recs = acct.run(ev, pb, li, &sel_cands, inc, sink);
            let mut fit: Vec<Option<Fitness>> = vec![None; us.len()];
            for (slot, (r, cached)) in recs.into_iter().enumerate() {
                let (u, sp) = match sel.get(slot) {
                    Some(&k) => {
                        fit[k] = Some(r.fitness());
                        (us[k].clone(), space.clone())
                    }
                    None => (free_u[slot - sel.len()].clone(), "free".to_string()),
                };
                let e = SearchEntry { eval_index, level: pb.levels[li].level, generation, u, cached, record: r, space: sp };
                eval_index += 1;
                sink.on_eval(&e);
                level_entries.push(e);
            }
            opt.tell_partial(&us, &fit);
            generation += 1;
            final_state = opt.state();
            sink.on_generation(&serde_json::json!({"generation": generation, "level": pb.levels[li].level,
                "sims": acct.sims, "cell_updates": acct.cell_updates, "optimizer": final_state}));
            // Guard against a search that only produces cached/static candidates forever.
            if acct.sims == sims_before && generation > 20 * (used_before + level_budget + 1) {
                stopped_by = "no_new_candidates".into();
                break 'levels;
            }
        }
        used_before = acct.sims;
        // Top-k distinct feasible-first candidates seed the next level.
        let mut ranked: Vec<&SearchEntry> = level_entries.iter().filter(|e| e.record.status != Status::StaticInfeasible && e.space == space).collect();
        ranked.sort_by(|a, b| a.record.fitness().cmp_deb(&b.record.fitness()).then(a.eval_index.cmp(&b.eval_index)));
        let mut picked: Vec<Vec<f64>> = vec![];
        let mut keys: Vec<&str> = vec![];
        for e in ranked {
            if !keys.contains(&e.record.key.as_str()) {
                keys.push(&e.record.key);
                picked.push(e.u.clone());
            }
            if picked.len() >= 4 {
                break;
            }
        }
        if let Some(u) = picked.first() {
            mean = u.clone();
            sigma = (sigma * 0.5).max(0.02);
        }
        seeds = picked;
        all.extend(level_entries);
    }
    // Final ranking only at full resolution (last level, which is 1).
    let full = pb.levels.last().unwrap().level;
    let finals: Vec<SearchEntry> = all.iter().filter(|e| e.level == full).cloned().collect();
    let best = finals
        .iter()
        .filter(|e| e.record.status == Status::Ok)
        .min_by(|a, b| a.record.fitness().cmp_deb(&b.record.fitness()).then(a.eval_index.cmp(&b.eval_index)))
        .cloned();
    SearchResult {
        method: oc.method.clone(),
        best,
        pareto: pareto(&finals),
        sims: acct.sims,
        cache_hits: acct.hits,
        cell_updates: acct.cell_updates,
        evaluations: eval_index,
        levels: pb.levels.iter().map(|l| l.level).collect(),
        stopped_by,
        final_state,
        utilization: ev.utilization(),
    }
}

pub fn cmp_entries(a: &SearchEntry, b: &SearchEntry) -> Ordering {
    a.record.fitness().cmp_deb(&b.record.fitness())
}
