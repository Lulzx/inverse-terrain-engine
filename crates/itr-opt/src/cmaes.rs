//! In-house CMA-ES (spec §6.4), following Hansen's tutorial (2016), in the normalized
//! box [0,1]^n with reflection repair. Jacobi eigendecomposition (n ≤ 64) uses only
//! + − × ÷ √, so the trace is platform-independent (§7.9).

use crate::rng::Rand;
use crate::search::{Fitness, Optimizer};
use serde_json::json;

pub struct CmaEs {
    n: usize,
    lambda: usize,
    mu: usize,
    w: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,
    pub mean: Vec<f64>,
    pub sigma: f64,
    c: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
    pc: Vec<f64>,
    ps: Vec<f64>,
    generation: usize,
    rng: Rand,
    sigma0: f64,
    restarts: usize,
    best_hist: Vec<f64>,
    /// lq-CMA-ES surrogate (Hansen 2019), when enabled.
    pub lq: Option<Lq>,
    /// Physics-informed restart means (e.g. one per flow corridor, §7.11.4), cycled.
    pub restart_means: Vec<Vec<f64>>,
}

/// Linear-quadratic surrogate state: archive of evaluated feasible points, current model,
/// and the adaptive number of candidates simulated per generation.
#[derive(Clone, Debug, Default)]
pub struct Lq {
    pub archive: Vec<(Vec<f64>, f64)>,
    pub coef: Option<Vec<f64>>,
    pub quadratic: bool,
    pub k_eval: usize,
    pub last_pred: Vec<f64>,
    pub tau_hist: Vec<f64>,
}

fn features(x: &[f64], quadratic: bool) -> Vec<f64> {
    let mut f = Vec::with_capacity(1 + 2 * x.len());
    f.push(1.0);
    f.extend_from_slice(x);
    if quadratic {
        f.extend(x.iter().map(|v| v * v));
    }
    f
}

/// Ridge least squares by Gaussian elimination with partial pivoting (+ − × ÷ only).
fn lstsq(rows: &[Vec<f64>], y: &[f64], ridge: f64) -> Option<Vec<f64>> {
    let m = rows.first()?.len();
    let mut a = vec![0.0; m * m];
    let mut b = vec![0.0; m];
    for (r, yi) in rows.iter().zip(y) {
        for i in 0..m {
            b[i] += r[i] * yi;
            for j in 0..m {
                a[i * m + j] += r[i] * r[j];
            }
        }
    }
    for i in 0..m {
        a[i * m + i] += ridge;
    }
    for c in 0..m {
        let piv = (c..m).max_by(|&p, &q| a[p * m + c].abs().total_cmp(&a[q * m + c].abs()))?;
        if a[piv * m + c].abs() < 1e-300 {
            return None;
        }
        if piv != c {
            for j in 0..m {
                a.swap(c * m + j, piv * m + j);
            }
            b.swap(c, piv);
        }
        for r in c + 1..m {
            let f = a[r * m + c] / a[c * m + c];
            if f != 0.0 {
                for j in c..m {
                    a[r * m + j] -= f * a[c * m + j];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; m];
    for c in (0..m).rev() {
        let mut s = b[c];
        for j in c + 1..m {
            s -= a[c * m + j] * x[j];
        }
        x[c] = s / a[c * m + c];
    }
    Some(x)
}

/// Kendall rank correlation τ_a.
fn kendall(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 1.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            s += ((a[i] - a[j]) * (b[i] - b[j])).signum();
        }
    }
    s / (n * (n - 1) / 2) as f64
}

impl Lq {
    fn refit(&mut self, n: usize, lambda: usize) {
        // Most recent points; quadratic (diagonal) model once there are enough of them.
        let quad = self.archive.len() >= 2 * (2 * n + 1);
        let need = if quad { 2 * n + 1 } else { n + 2 };
        if self.archive.len() < need.max(lambda.min(4)) {
            self.coef = None;
            return;
        }
        let keep = self.archive.len().min((20 * n).max(4 * need));
        let pts = &self.archive[self.archive.len() - keep..];
        let rows: Vec<Vec<f64>> = pts.iter().map(|(x, _)| features(x, quad)).collect();
        let y: Vec<f64> = pts.iter().map(|(_, f)| *f).collect();
        self.quadratic = quad;
        self.coef = lstsq(&rows, &y, 1e-10);
    }
    fn predict(&self, x: &[f64]) -> Option<f64> {
        let c = self.coef.as_ref()?;
        Some(features(x, self.quadratic).iter().zip(c).map(|(a, b)| a * b).sum())
    }
}

/// Symmetric eigendecomposition by cyclic Jacobi rotations. Returns (eigenvalues,
/// eigenvectors column-major in `v`, i.e. v[r*n + k] is row r of eigenvector k).
pub fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}

fn reflect(x: f64) -> f64 {
    // Reflect into [0,1] (period 2).
    let mut y = x % 2.0;
    if y < 0.0 {
        y += 2.0;
    }
    if y > 1.0 { 2.0 - y } else { y }
}

impl CmaEs {
    pub fn new(n: usize, lambda: Option<usize>, mean: Vec<f64>, sigma: f64, seed: u64) -> Self {
        let lambda = lambda.unwrap_or(4 + (3.0 * libm::log(n as f64)).floor() as usize).max(2);
        let mut s = Self::empty(n, lambda, mean, sigma, seed);
        s.sigma0 = sigma;
        s
    }

    fn empty(n: usize, lambda: usize, mean: Vec<f64>, sigma: f64, seed: u64) -> Self {
        let mu = lambda / 2;
        let mut w: Vec<f64> = (0..mu).map(|i| libm::log(mu as f64 + 0.5) - libm::log(i as f64 + 1.0)).collect();
        let sw: f64 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= sw);
        let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
        let nf = n as f64;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let c1 = 2.0 / ((nf + 1.3) * (nf + 1.3) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0) * (nf + 2.0) + mueff));
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        let mut c = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            c[i * n + i] = 1.0;
            b[i * n + i] = 1.0;
        }
        Self {
            n,
            lambda,
            mu,
            w,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            mean,
            sigma,
            c,
            b,
            d: vec![1.0; n],
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            generation: 0,
            rng: Rand::new(seed),
            sigma0: sigma,
            restarts: 0,
            best_hist: vec![],
            lq: None,
            restart_means: vec![],
        }
    }

    fn sample_raw(&mut self) -> Vec<f64> {
        let n = self.n;
        let z: Vec<f64> = (0..n).map(|_| self.rng.normal()).collect();
        let mut x = self.mean.clone();
        for r in 0..n {
            let mut y = 0.0;
            for k in 0..n {
                y += self.b[r * n + k] * self.d[k] * z[k];
            }
            x[r] = reflect(x[r] + self.sigma * y);
        }
        x
    }

    fn update(&mut self, xs: &[Vec<f64>], order: &[usize]) {
        let n = self.n;
        let old = self.mean.clone();
        let mu = self.mu.min(order.len());
        let mut new_mean = vec![0.0; n];
        let wsum: f64 = self.w[..mu].iter().sum();
        for (k, &idx) in order.iter().take(mu).enumerate() {
            for i in 0..n {
                new_mean[i] += self.w[k] / wsum * xs[idx][i];
            }
        }
        self.mean = new_mean;
        let yw: Vec<f64> = (0..n).map(|i| (self.mean[i] - old[i]) / self.sigma).collect();
        // C^{-1/2} yw = B D^{-1} B^T yw
        let mut bt = vec![0.0; n];
        for k in 0..n {
            let mut s = 0.0;
            for r in 0..n {
                s += self.b[r * n + k] * yw[r];
            }
            bt[k] = s / self.d[k];
        }
        let mut cinv = vec![0.0; n];
        for r in 0..n {
            for k in 0..n {
                cinv[r] += self.b[r * n + k] * bt[k];
            }
        }
        let csn = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for i in 0..n {
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + csn * cinv[i];
        }
        let psn = self.ps.iter().map(|x| x * x).sum::<f64>().sqrt();
        let gen1 = (self.generation + 1) as f64;
        let hsig = psn / (1.0 - libm::pow(1.0 - self.cs, 2.0 * gen1)).sqrt() / self.chi_n < 1.4 + 2.0 / (n as f64 + 1.0);
        let ccn = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        for i in 0..n {
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + if hsig { ccn * yw[i] } else { 0.0 };
        }
        let dh = if hsig { 0.0 } else { self.cc * (2.0 - self.cc) };
        let ys: Vec<Vec<f64>> = order.iter().take(mu).map(|&idx| (0..n).map(|i| (xs[idx][i] - old[i]) / self.sigma).collect()).collect();
        for r in 0..n {
            for c in 0..n {
                let mut rank_mu = 0.0;
                for (k, y) in ys.iter().enumerate() {
                    rank_mu += self.w[k] / wsum * y[r] * y[c];
                }
                self.c[r * n + c] = (1.0 - self.c1 - self.cmu) * self.c[r * n + c]
                    + self.c1 * (self.pc[r] * self.pc[c] + dh * self.c[r * n + c])
                    + self.cmu * rank_mu;
            }
        }
        self.sigma *= libm::exp((self.cs / self.damps) * (psn / self.chi_n - 1.0));
        self.sigma = self.sigma.min(1.0);
        let (ev, v) = jacobi_eigen(&self.c, n);
        self.b = v;
        self.d = ev.iter().map(|x| x.max(1e-20).sqrt()).collect();
        self.generation += 1;
    }
}


impl CmaEs {
    fn tell_ordered(&mut self, xs: &[Vec<f64>], order: &[usize], best: f64) {
        self.update(xs, order);
        self.best_hist.push(best);
        // IPOP-style restart on collapse or stagnation: doubled population, fresh mean.
        let cond = self.d.iter().cloned().fold(0.0, f64::max) / self.d.iter().cloned().fold(f64::MAX, f64::min);
        let stagnant = self.best_hist.len() > 10 + 30 * self.n / self.lambda
            && self.best_hist[self.best_hist.len() - 1] >= self.best_hist[self.best_hist.len() - 1 - 10] - 1e-12;
        if self.sigma < 1e-4 || cond > 1e7 || stagnant {
            self.restarts += 1;
            let seed = self.rng.uniform().to_bits();
            let mean: Vec<f64> = if self.restart_means.is_empty() {
                (0..self.n).map(|_| self.rng.uniform()).collect()
            } else {
                self.restart_means[self.restarts % self.restart_means.len()].clone()
            };
            let lam = self.lambda * 2;
            let (r, rm) = (self.restarts, std::mem::take(&mut self.restart_means));
            let lq = self.lq.take().map(|l| Lq { k_eval: 0, ..l });
            *self = Self::empty(self.n, lam, mean, self.sigma0, seed);
            self.restarts = r;
            self.restart_means = rm;
            self.lq = lq;
        }
    }
}

impl Optimizer for CmaEs {
    fn name(&self) -> &'static str {
        "cma_es"
    }
    fn batch_size(&self) -> usize {
        self.lambda
    }
    fn propose(&mut self, _slot: usize, _attempt: usize) -> Option<Vec<f64>> {
        Some(self.sample_raw())
    }
    fn select(&mut self, xs: &[Vec<f64>]) -> Vec<bool> {
        let n = self.n;
        let lambda = self.lambda;
        let Some(lq) = self.lq.as_mut() else { return vec![true; xs.len()] };
        lq.refit(n, lambda);
        let preds: Option<Vec<f64>> = xs.iter().map(|x| lq.predict(x)).collect();
        let Some(preds) = preds else {
            lq.last_pred.clear();
            return vec![true; xs.len()];
        };
        if lq.k_eval == 0 {
            lq.k_eval = (lambda / 2).max(2);
        }
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| preds[a].total_cmp(&preds[b]).then(a.cmp(&b)));
        let mut mask = vec![false; xs.len()];
        for &k in order.iter().take(lq.k_eval.min(xs.len())) {
            mask[k] = true;
        }
        lq.last_pred = preds;
        mask
    }

    fn tell_partial(&mut self, xs: &[Vec<f64>], fit: &[Option<Fitness>]) {
        let Some(lq) = self.lq.as_mut() else {
            let f: Vec<Fitness> = fit.iter().map(|f| f.expect("all evaluated")).collect();
            return self.tell(xs, &f);
        };
        for (x, f) in xs.iter().zip(fit) {
            if let Some(f) = f.filter(|f| f.feasible) {
                lq.archive.push((x.clone(), f.j));
            }
        }
        if lq.last_pred.len() != xs.len() {
            // No model yet: everything was evaluated.
            let f: Vec<Fitness> = fit.iter().map(|f| f.expect("all evaluated")).collect();
            return self.tell(xs, &f);
        }
        // Model quality on the simulated ones → adapt how many to simulate next time.
        let (mut tp, mut tt) = (vec![], vec![]);
        for (k, f) in fit.iter().enumerate() {
            if let Some(f) = f.filter(|f| f.feasible) {
                tp.push(lq.last_pred[k]);
                tt.push(f.j);
            }
        }
        let tau = kendall(&tp, &tt);
        lq.tau_hist.push(tau);
        lq.k_eval = if tau < 0.85 { (lq.k_eval * 2).min(self.lambda) } else { ((lq.k_eval * 3).div_ceil(4)).max(2) };
        // Ranking: simulated feasible by true J, then unsimulated by prediction, then
        // simulated infeasible by Deb's rules.
        let preds = lq.last_pred.clone();
        let mut order: Vec<usize> = (0..xs.len()).collect();
        let class = |k: usize| match fit[k] {
            Some(f) if f.feasible => 0,
            None => 1,
            Some(_) => 2,
        };
        order.sort_by(|&a, &b| {
            class(a).cmp(&class(b)).then_with(|| match (fit[a], fit[b]) {
                (Some(x), Some(y)) => x.cmp_deb(&y),
                _ => preds[a].total_cmp(&preds[b]),
            }).then(a.cmp(&b))
        });
        let best = fit.iter().flatten().map(|f| f.rank_value()).fold(f64::MAX, f64::min);
        self.tell_ordered(xs, &order, best);
    }

    fn tell(&mut self, xs: &[Vec<f64>], fit: &[Fitness]) {
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| fit[a].cmp_deb(&fit[b]).then(a.cmp(&b)));
        let best = fit[order[0]].rank_value();
        self.tell_ordered(xs, &order, best);
    }

    fn state(&self) -> serde_json::Value {
        json!({"method": if self.lq.is_some() { "lq_cma_es" } else { "cma_es" }, "generation": self.generation, "lambda": self.lambda,
               "sigma": self.sigma, "mean": self.mean, "restarts": self.restarts,
               "lq": self.lq.as_ref().map(|l| json!({"k_eval": l.k_eval, "archive": l.archive.len(), "quadratic": l.quadratic,
                   "tau_last": l.tau_hist.last()}))})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Fitness;

    #[test]
    fn jacobi_reconstructs() {
        let a = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let (e, v) = jacobi_eigen(&a, 3);
        for r in 0..3 {
            for c in 0..3 {
                let s: f64 = (0..3).map(|k| v[r * 3 + k] * e[k] * v[c * 3 + k]).sum();
                assert!((s - a[r * 3 + c]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn minimizes_sphere_in_box() {
        let n = 6;
        let mut es = CmaEs::new(n, None, vec![0.9; n], 0.3, 1);
        let target = [0.3, 0.6, 0.2, 0.8, 0.5, 0.4];
        let mut best = f64::MAX;
        for _ in 0..150 {
            let xs: Vec<Vec<f64>> = (0..es.batch_size()).map(|k| es.propose(k, 0).unwrap()).collect();
            let fit: Vec<Fitness> = xs.iter().map(|x| Fitness::feasible(x.iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum())).collect();
            for f in &fit {
                best = best.min(f.rank_value());
            }
            es.tell(&xs, &fit);
        }
        assert!(best < 1e-6, "best {best}");
    }
}

#[cfg(test)]
mod lq_tests {
    use super::*;
    use crate::search::{Fitness, Optimizer};

    /// lq-CMA-ES reaches the same target on a smooth (quadratic) function with fewer
    /// true evaluations than plain CMA-ES (spec §19.2).
    #[test]
    fn surrogate_saves_evaluations() {
        let n = 8;
        let target: Vec<f64> = (0..n).map(|i| 0.2 + 0.07 * i as f64).collect();
        let f = |x: &[f64]| x.iter().zip(&target).enumerate().map(|(i, (a, b))| (1.0 + i as f64) * (a - b) * (a - b)).sum::<f64>();
        let run = |lq: bool| {
            let mut es = CmaEs::new(n, None, vec![0.5; n], 0.3, 3);
            if lq {
                es.lq = Some(Lq::default());
            }
            let mut evals = 0;
            for _ in 0..400 {
                let xs: Vec<Vec<f64>> = (0..es.batch_size()).map(|k| es.propose(k, 0).unwrap()).collect();
                let mask = es.select(&xs);
                let fit: Vec<Option<Fitness>> = xs.iter().zip(&mask).map(|(x, &m)| m.then(|| Fitness::feasible(f(x)))).collect();
                evals += mask.iter().filter(|m| **m).count();
                if fit.iter().flatten().any(|v| v.j < 1e-8) {
                    return evals;
                }
                es.tell_partial(&xs, &fit);
            }
            usize::MAX
        };
        let (plain, lq) = (run(false), run(true));
        println!("evaluations to 1e-8: plain {plain}, lq {lq}");
        assert!(lq < plain, "lq {lq} vs plain {plain}");
    }
}
