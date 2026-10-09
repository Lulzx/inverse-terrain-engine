//! Ablation / Shapley attribution of the final design (spec §19.2): evaluate every subset
//! of the active primitives (inactive ones get zero height) and attribute the objective
//! improvement to each primitive with exact Shapley values. Shows which primitives carry
//! the design and which are redundant.

use crate::evaluator::Evaluator;
use crate::problem::Problem;
use itr_core::design::{Primitive, PrimitiveKind};
use itr_core::Real;
use serde::Serialize;

/// Exact enumeration up to this many active primitives (2^K simulations); above it only
/// leave-one-out and single-primitive runs are made (2K + 2).
pub const MAX_EXACT: usize = 8;

#[derive(Clone, Debug, Serialize)]
pub struct SubsetResult {
    /// Indices into the design's primitive list.
    pub primitives: Vec<usize>,
    pub j: f64,
    pub feasible: bool,
    pub assets_exceeding: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrimitiveAttribution {
    pub index: usize,
    pub kind: PrimitiveKind,
    /// Shapley value of the J reduction (positive = helps). Exact when `exact`.
    pub shapley_j_reduction: Option<f64>,
    /// J(all) − J(all without this primitive), negated: J increase when removed.
    pub leave_one_out_j_increase: f64,
    /// J(none) − J(only this primitive).
    pub alone_j_reduction: f64,
    /// Removing it changes J by less than 1% of the total improvement and keeps feasibility.
    pub redundant: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Ablation {
    pub level: usize,
    pub exact: bool,
    pub simulations: usize,
    pub j_none: f64,
    pub j_all: f64,
    pub attributions: Vec<PrimitiveAttribution>,
    pub subsets: Vec<SubsetResult>,
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Ablation of `prims` at level index `li` (the finest search level).
pub fn ablate<T: Real>(pb: &Problem<T>, ev: &mut Evaluator<T>, li: usize, prims: &[Primitive]) -> Ablation {
    let active: Vec<usize> = (0..prims.len()).filter(|&i| prims[i].height != 0.0).collect();
    let k = active.len();
    let exact = k <= MAX_EXACT;
    // Subsets as bit masks over `active`.
    let masks: Vec<u32> = if exact {
        (0..1u32 << k).collect()
    } else {
        let full = (1u32 << k) - 1;
        let mut m = vec![0, full];
        for b in 0..k {
            m.push(1 << b);
            m.push(full & !(1 << b));
        }
        m
    };
    let cands: Vec<_> = masks
        .iter()
        .map(|&m| {
            let mut ps = prims.to_vec();
            for (b, &i) in active.iter().enumerate() {
                if m >> b & 1 == 0 {
                    ps[i].height = 0.0;
                }
            }
            Evaluator::prepare_prims(pb, li, ps)
        })
        .collect();
    let sims0 = ev.sims;
    let recs = ev.eval_batch(pb, li, &cands, None);
    let j_of = |m: u32| recs[masks.iter().position(|&x| x == m).unwrap()].j;
    let full = if k == 0 { 0 } else { (1u32 << k) - 1 };
    let (j_none, j_all) = (j_of(0), j_of(full));
    let total = j_none - j_all;
    let attributions = active
        .iter()
        .enumerate()
        .map(|(b, &i)| {
            let bit = 1u32 << b;
            let shapley = exact.then(|| {
                masks
                    .iter()
                    .filter(|&&m| m & bit == 0)
                    .map(|&m| {
                        let s = m.count_ones() as usize;
                        factorial(s) * factorial(k - s - 1) / factorial(k) * (j_of(m) - j_of(m | bit))
                    })
                    .sum::<f64>()
            });
            let loo = j_of(full & !bit) - j_all;
            let loo_feasible = recs[masks.iter().position(|&x| x == full & !bit).unwrap()].fitness_feasible;
            PrimitiveAttribution {
                index: i,
                kind: prims[i].kind,
                shapley_j_reduction: shapley,
                leave_one_out_j_increase: loo,
                alone_j_reduction: j_none - j_of(bit),
                redundant: loo_feasible && loo.abs() < 0.01 * total.abs().max(1e-12),
            }
        })
        .collect();
    let subsets = masks
        .iter()
        .zip(&recs)
        .map(|(&m, r)| SubsetResult {
            primitives: active.iter().enumerate().filter(|(b, _)| m >> b & 1 == 1).map(|(_, &i)| i).collect(),
            j: r.j,
            feasible: r.fitness_feasible,
            assets_exceeding: r.report.as_ref().map(|x| x.assets_exceeding),
        })
        .collect();
    Ablation { level: pb.levels[li].level, exact, simulations: ev.sims - sims0, j_none, j_all, attributions, subsets }
}
