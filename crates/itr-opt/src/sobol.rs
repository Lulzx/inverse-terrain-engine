//! Scrambled Sobol sequence (spec §19.2) for initial designs and the quasi-random
//! baseline. Dependency-free:
//! - primitive polynomials over GF(2) are enumerated in-house (degree order, tested by
//!   the multiplicative order of x);
//! - initial direction numbers are odd `m_k < 2^k` drawn from the pinned ChaCha8 stream
//!   (every such choice gives a valid Sobol (t,s)-sequence; Joe–Kuo's optimized tables
//!   are not reproduced here);
//! - each point gets a random linear matrix scramble plus a digital shift (Matoušek),
//!   seeded, so the sequence is deterministic per seed.

use crate::rng::Rand;

const BITS: usize = 32;

fn polymulmod(a: u64, b: u64, p: u64, deg: u32) -> u64 {
    let mut r = 0u64;
    let mut a = a;
    let mut b = b;
    while b != 0 {
        if b & 1 == 1 {
            r ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a >> deg & 1 == 1 {
            a ^= p;
        }
    }
    r
}

fn polypow(e: u64, p: u64, deg: u32) -> u64 {
    let (mut base, mut r, mut e) = (2u64, 1u64, e);
    while e > 0 {
        if e & 1 == 1 {
            r = polymulmod(r, base, p, deg);
        }
        base = polymulmod(base, base, p, deg);
        e >>= 1;
    }
    r
}

fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut v = vec![];
    let mut f = 2;
    while f * f <= n {
        if n.is_multiple_of(f) {
            v.push(f);
            while n.is_multiple_of(f) {
                n /= f;
            }
        }
        f += 1;
    }
    if n > 1 {
        v.push(n);
    }
    v
}

/// Primitive polynomials (bit i = coefficient of x^i), degree ≥ 1, in increasing order.
pub fn primitive_polynomials(count: usize) -> Vec<(u32, u64)> {
    let mut out = vec![];
    let mut deg = 1u32;
    while out.len() < count {
        let ord = (1u64 << deg) - 1;
        let fac = prime_factors(ord);
        for p in (1u64 << deg)..(1u64 << (deg + 1)) {
            if p & 1 == 0 {
                continue;
            }
            if polypow(ord, p, deg) == 1 && fac.iter().all(|q| polypow(ord / q, p, deg) != 1) {
                out.push((deg, p));
                if out.len() == count {
                    break;
                }
            }
        }
        deg += 1;
    }
    out
}

pub struct Sobol {
    dim: usize,
    v: Vec<[u32; BITS]>,
    x: Vec<u32>,
    index: u64,
    scramble: Vec<[u32; BITS]>,
    shift: Vec<u32>,
}

impl Sobol {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = Rand::new(seed ^ 0x50B0_1000);
        let polys = primitive_polynomials(dim.saturating_sub(1));
        let mut v = vec![[0u32; BITS]; dim];
        for k in 0..BITS {
            v[0][k] = 1 << (BITS - 1 - k);
        }
        for d in 1..dim {
            let (s, p) = polys[d - 1];
            let s = s as usize;
            let mut m = [0u32; BITS];
            for (k, mk) in m.iter_mut().enumerate().take(s) {
                // Odd m_k in [1, 2^(k+1)).
                let r = (rng.uniform() * (1u64 << k) as f64) as u32;
                *mk = 2 * r + 1;
            }
            for k in s..BITS {
                let mut val = m[k - s] ^ (m[k - s] << s);
                for i in 1..s {
                    if (p >> (s - i)) & 1 == 1 {
                        val ^= m[k - i] << i;
                    }
                }
                m[k] = val;
            }
            for k in 0..BITS {
                v[d][k] = m[k] << (BITS - 1 - k);
            }
        }
        // Random lower-triangular (unit diagonal) scramble matrices and digital shifts.
        let scramble = (0..dim)
            .map(|_| {
                let mut rows = [0u32; BITS];
                for (r, row) in rows.iter_mut().enumerate() {
                    let mut bits = 1u32 << (BITS - 1 - r);
                    for c in 0..r {
                        if rng.uniform() < 0.5 {
                            bits |= 1 << (BITS - 1 - c);
                        }
                    }
                    *row = bits;
                }
                rows
            })
            .collect();
        let shift = (0..dim).map(|_| (rng.uniform() * 4294967296.0) as u32).collect();
        Self { dim, v, x: vec![0; dim], index: 0, scramble, shift }
    }

    /// Next point in [0,1)^dim (the unscrambled first point is the origin; scrambling
    /// moves it, so every point is usable).
    pub fn next_point(&mut self) -> Vec<f64> {
        if self.index > 0 {
            let c = (self.index - 1).trailing_ones() as usize;
            for d in 0..self.dim {
                self.x[d] ^= self.v[d][c.min(BITS - 1)];
            }
        }
        self.index += 1;
        (0..self.dim)
            .map(|d| {
                let x = self.x[d];
                // y = L·x (bit-matrix product over GF(2)), then digital shift.
                let mut y = 0u32;
                for (r, row) in self.scramble[d].iter().enumerate() {
                    if (row & x).count_ones() & 1 == 1 {
                        y |= 1 << (BITS - 1 - r);
                    }
                }
                ((y ^ self.shift[d]) as f64 + 0.5) / 4294967296.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_primitive_polynomials() {
        // x+1, x²+x+1, x³+x+1, x³+x²+1, x⁴+x+1, x⁴+x³+1
        let p = primitive_polynomials(6);
        assert_eq!(p.iter().map(|x| x.1).collect::<Vec<_>>(), vec![0b11, 0b111, 0b1011, 0b1101, 0b10011, 0b11001]);
    }

    #[test]
    fn stratified_and_deterministic() {
        // First 2^k points of a (scrambled) Sobol sequence put exactly one point in each
        // dyadic interval of length 2^-k, in every coordinate.
        let dim = 12;
        let mut s = Sobol::new(dim, 7);
        let pts: Vec<Vec<f64>> = (0..64).map(|_| s.next_point()).collect();
        for d in 0..dim {
            let mut seen = [false; 64];
            for p in &pts {
                let b = (p[d] * 64.0) as usize;
                assert!(!seen[b], "dim {d} bin {b} hit twice");
                seen[b] = true;
            }
        }
        let mut t = Sobol::new(dim, 7);
        assert_eq!(t.next_point(), pts[0]);
    }
}
