//! Optimizer interface and baselines (spec §6.4): seeded random search and compass
//! (coordinate) search. All operate in the normalized box [0,1]^n.

use crate::rng::Rand;
use serde_json::json;
use std::cmp::Ordering;

/// Fitness with feasibility-first (Deb's rules) ordering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fitness {
    pub feasible: bool,
    /// Objective J (lower is better); for infeasible, the objective lower bound.
    pub j: f64,
    /// Constraint violation magnitude (0 when feasible).
    pub violation: f64,
    /// Guard violation: sync index of first breach (later is better); 0 otherwise.
    pub breach_sync: u32,
}

impl Fitness {
    pub fn feasible(j: f64) -> Self {
        Self { feasible: true, j, violation: 0.0, breach_sync: 0 }
    }
    pub fn infeasible(violation: f64, breach_sync: u32, j: f64) -> Self {
        Self { feasible: false, j, violation, breach_sync }
    }
    /// Deb's rules: feasible before infeasible; feasible by J; infeasible by smaller
    /// violation, then later breach.
    pub fn cmp_deb(&self, o: &Fitness) -> Ordering {
        match (self.feasible, o.feasible) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => self.j.total_cmp(&o.j),
            (false, false) => self.violation.total_cmp(&o.violation).then(o.breach_sync.cmp(&self.breach_sync)),
        }
    }
    /// Scalar for stagnation tracking: J if feasible, else large + violation.
    pub fn rank_value(&self) -> f64 {
        if self.feasible { self.j } else { 1e12 + self.violation }
    }
}

pub trait Optimizer {
    fn name(&self) -> &'static str;
    fn batch_size(&self) -> usize;
    /// Proposal for `slot` at resampling `attempt` (None = cannot resample).
    fn propose(&mut self, slot: usize, attempt: usize) -> Option<Vec<f64>>;
    fn tell(&mut self, xs: &[Vec<f64>], fit: &[Fitness]);
    fn state(&self) -> serde_json::Value;
    /// Elitist methods may abort candidates against a frozen incumbent (§7.6).
    fn incumbent(&self) -> Option<f64> {
        None
    }
    /// Which proposals to simulate (surrogate pre-screening, lq-CMA-ES). Default: all.
    fn select(&mut self, xs: &[Vec<f64>]) -> Vec<bool> {
        vec![true; xs.len()]
    }
    /// Tell with only the selected proposals evaluated (`None` = not simulated).
    fn tell_partial(&mut self, xs: &[Vec<f64>], fit: &[Option<Fitness>]) {
        let (x, f): (Vec<Vec<f64>>, Vec<Fitness>) = xs.iter().zip(fit).filter_map(|(x, f)| f.map(|f| (x.clone(), f))).unzip();
        self.tell(&x, &f);
    }
}

/// Quasi-random baseline: scrambled Sobol points (spec §19.2).
pub struct SobolSearch {
    lambda: usize,
    seq: crate::sobol::Sobol,
    best: Option<f64>,
}

impl SobolSearch {
    pub fn new(n: usize, lambda: usize, seed: u64) -> Self {
        Self { lambda, seq: crate::sobol::Sobol::new(n, seed), best: None }
    }
}

impl Optimizer for SobolSearch {
    fn name(&self) -> &'static str {
        "sobol"
    }
    fn batch_size(&self) -> usize {
        self.lambda
    }
    fn propose(&mut self, _slot: usize, _attempt: usize) -> Option<Vec<f64>> {
        Some(self.seq.next_point())
    }
    fn tell(&mut self, _xs: &[Vec<f64>], fit: &[Fitness]) {
        for f in fit.iter().filter(|f| f.feasible) {
            if self.best.is_none_or(|b| f.j < b) {
                self.best = Some(f.j);
            }
        }
    }
    fn state(&self) -> serde_json::Value {
        json!({"method": "sobol", "best": self.best})
    }
    fn incumbent(&self) -> Option<f64> {
        self.best
    }
}

pub struct RandomSearch {
    n: usize,
    lambda: usize,
    rng: Rand,
    best: Option<f64>,
}

impl RandomSearch {
    pub fn new(n: usize, lambda: usize, seed: u64) -> Self {
        Self { n, lambda, rng: Rand::new(seed), best: None }
    }
}

impl Optimizer for RandomSearch {
    fn name(&self) -> &'static str {
        "random"
    }
    fn batch_size(&self) -> usize {
        self.lambda
    }
    fn propose(&mut self, _slot: usize, _attempt: usize) -> Option<Vec<f64>> {
        Some((0..self.n).map(|_| self.rng.uniform()).collect())
    }
    fn tell(&mut self, _xs: &[Vec<f64>], fit: &[Fitness]) {
        for f in fit.iter().filter(|f| f.feasible) {
            if self.best.is_none_or(|b| f.j < b) {
                self.best = Some(f.j);
            }
        }
    }
    fn state(&self) -> serde_json::Value {
        json!({"method": "random", "best": self.best})
    }
    fn incumbent(&self) -> Option<f64> {
        self.best
    }
}

/// Compass search: evaluate ±step along every coordinate; move to the best improving
/// point, otherwise halve the step.
pub struct CompassSearch {
    n: usize,
    center: Vec<f64>,
    center_fit: Option<Fitness>,
    step: f64,
    pending_center: bool,
}

impl CompassSearch {
    pub fn new(center: Vec<f64>, step: f64) -> Self {
        Self { n: center.len(), center, center_fit: None, step, pending_center: true }
    }
    fn point(&self, slot: usize) -> Vec<f64> {
        let mut p = self.center.clone();
        if !self.pending_center {
            let (k, sgn) = (slot / 2, if slot.is_multiple_of(2) { 1.0 } else { -1.0 });
            p[k] = (p[k] + sgn * self.step).clamp(0.0, 1.0);
        }
        p
    }
}

impl Optimizer for CompassSearch {
    fn name(&self) -> &'static str {
        "coordinate"
    }
    fn batch_size(&self) -> usize {
        if self.pending_center { 1 } else { 2 * self.n }
    }
    fn propose(&mut self, slot: usize, attempt: usize) -> Option<Vec<f64>> {
        if attempt > 0 { None } else { Some(self.point(slot)) }
    }
    fn tell(&mut self, xs: &[Vec<f64>], fit: &[Fitness]) {
        if self.pending_center {
            self.center_fit = Some(fit[0]);
            self.pending_center = false;
            return;
        }
        let mut best = 0;
        for k in 1..fit.len() {
            if fit[k].cmp_deb(&fit[best]) == Ordering::Less {
                best = k;
            }
        }
        let improves = self.center_fit.is_none_or(|c| fit[best].cmp_deb(&c) == Ordering::Less);
        if improves {
            self.center = xs[best].clone();
            self.center_fit = Some(fit[best]);
        } else {
            self.step *= 0.5;
            if self.step < 1e-3 {
                self.step = 0.25;
            }
        }
    }
    fn state(&self) -> serde_json::Value {
        json!({"method": "coordinate", "center": self.center, "step": self.step})
    }
    fn incumbent(&self) -> Option<f64> {
        self.center_fit.filter(|f| f.feasible).map(|f| f.j)
    }
}
