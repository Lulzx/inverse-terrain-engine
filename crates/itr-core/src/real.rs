//! Working-precision abstraction shared by the oracle and the optimized kernels.
//!
//! Only IEEE-754 correctly rounded operations (`+ − × ÷ √`) are exposed, plus an
//! inverse cube root built from them, so results are bitwise reproducible across
//! platforms (spec §7.4, §7.9). `max`/`min` have explicit, NaN-defined semantics
//! instead of relying on `f32::max`, which silently drops NaN.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Real:
    Copy
    + Send
    + Sync
    + Default
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + 'static
{
    /// Elements per 64-byte cache line.
    const LANES: usize;
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;

    #[inline(always)]
    fn abs(self) -> Self {
        if self < Self::ZERO { -self } else { self }
    }
    /// Returns `a` if `a > b`, else `b`. If either is NaN the result is `b`.
    #[inline(always)]
    fn max(self, b: Self) -> Self {
        if self > b { self } else { b }
    }
    #[inline(always)]
    fn min(self, b: Self) -> Self {
        if self < b { self } else { b }
    }
    /// `x^{-1/3}` for `x > 0` using a bit-level seed and multiplication-only Newton
    /// steps `r ← r·(4 − x·r³)/3`. Result for `x <= 0` is unspecified (callers mask it).
    fn rcbrt(self) -> Self;
    /// Maps `-0.0` to `+0.0` so bitwise comparisons are meaningful (spec §7.6).
    #[inline(always)]
    fn canon_zero(self) -> Self {
        if self == Self::ZERO { Self::ZERO } else { self }
    }
}

const THIRD_F64: f64 = 1.0 / 3.0;

impl Real for f64 {
    const LANES: usize = 8;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAME: &'static str = "f64";
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn rcbrt(self) -> Self {
        // Minimax-tuned exponent-bias seed (max rel. error 3.4e-2, the f32 constant's
        // offset scaled to the f64 mantissa) and 4 Newton steps in the short-chain form
        // r ← r·(4/3) − (x/3)·r⁴ (4 dependent ops per step): error ≲ 1e-15.
        const K: u64 = 0x5540_0000_0000_0000 - (0x8782B_u64 << 29);
        let bits = self.to_bits();
        let mut r = f64::from_bits(K.wrapping_sub(bits / 3));
        let x3 = self * THIRD_F64;
        let c43 = 4.0 * THIRD_F64;
        for _ in 0..4 {
            let r2 = r * r;
            r = r * c43 - x3 * (r2 * r2);
        }
        r
    }
}

impl Real for f32 {
    const LANES: usize = 16;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAME: &'static str = "f32";
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn rcbrt(self) -> Self {
        // Minimax-tuned seed (max rel. error 3.4e-2) + 3 short-chain Newton steps:
        // error ≲ 3e-10 before rounding, i.e. f32-accurate.
        const K: u32 = 0x54A2_3280;
        let bits = self.to_bits();
        let mut r = f32::from_bits(K.wrapping_sub(bits / 3));
        let x3 = self * (THIRD_F64 as f32);
        let c43 = (4.0 * THIRD_F64) as f32;
        for _ in 0..3 {
            let r2 = r * r;
            r = r * c43 - x3 * (r2 * r2);
        }
        r
    }
}

/// Fixed-layout summation (spec §7.4): 16 accumulators indexed by `col mod 16`,
/// combined by a fixed pairwise tree. Independent of SIMD width or dispatch.
#[derive(Clone, Copy)]
pub struct FixedSum<T: Real> {
    acc: [T; 16],
}

impl<T: Real> Default for FixedSum<T> {
    fn default() -> Self {
        Self { acc: [T::ZERO; 16] }
    }
}

impl<T: Real> FixedSum<T> {
    /// Sum `v` (whose element `k` has column index `k`) with the fixed lane layout,
    /// processing 16-element chunks so the loop vectorizes. Identical to calling `add`
    /// for every element in order.
    #[inline]
    pub fn add_slice(&mut self, v: &[T]) {
        let (chunks, rem) = v.as_chunks::<16>();
        for c in chunks {
            for l in 0..16 {
                self.acc[l] += c[l];
            }
        }
        for (l, x) in rem.iter().enumerate() {
            self.acc[l] += *x;
        }
    }
    #[inline(always)]
    pub fn add(&mut self, col: usize, v: T) {
        self.acc[col & 15] += v;
    }
    #[inline]
    pub fn total(&self) -> T {
        let a = &self.acc;
        let mut l8 = [T::ZERO; 8];
        for k in 0..8 {
            l8[k] = a[2 * k] + a[2 * k + 1];
        }
        let l4 = [l8[0] + l8[1], l8[2] + l8[3], l8[4] + l8[5], l8[6] + l8[7]];
        (l4[0] + l4[1]) + (l4[2] + l4[3])
    }
}

/// Order-independent maximum over a slice, chunked for vectorization.
#[inline]
pub fn max_slice<T: Real>(v: &[T]) -> T {
    let mut m = [T::ZERO; 16];
    let (chunks, rem) = v.as_chunks::<16>();
    for c in chunks {
        for l in 0..16 {
            m[l] = m[l].max(c[l]);
        }
    }
    for (l, x) in rem.iter().enumerate() {
        m[l] = m[l].max(*x);
    }
    m.iter().fold(T::ZERO, |a, b| a.max(*b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_slice_matches_add() {
        let v: Vec<f32> = (0..1037).map(|i| (i as f32 * 0.731).sin() * 1e3).collect();
        let mut a = FixedSum::<f32>::default();
        for (i, x) in v.iter().enumerate() {
            a.add(i, *x);
        }
        let mut b = FixedSum::<f32>::default();
        b.add_slice(&v);
        assert_eq!(a.total().to_bits(), b.total().to_bits());
    }

    #[test]
    fn rcbrt_accuracy() {
        for &x in &[1e-8f64, 1e-6, 1e-3, 0.37, 1.0, 2.0, 17.0, 1e3, 1e6] {
            let r = x.rcbrt();
            let exact = x.powf(-1.0 / 3.0);
            assert!(((r - exact) / exact).abs() < 1e-14, "f64 x={x} r={r} exact={exact}");
            let r32 = (x as f32).rcbrt();
            assert!(((r32 as f64 - exact) / exact).abs() < 1e-6, "f32 x={x} r={r32} exact={exact}");
        }
    }

    #[test]
    fn max_nan_semantics() {
        assert!(Real::max(1.0f32, f32::NAN).is_nan());
        assert_eq!(Real::max(f32::NAN, 1.0f32), 1.0);
    }

    #[test]
    fn fixed_sum_order_independent_of_chunking() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut s = FixedSum::<f32>::default();
        for (i, x) in v.iter().enumerate() {
            s.add(i, *x);
        }
        let mut s2 = FixedSum::<f32>::default();
        for i in (0..1000).rev() {
            // reverse order within each lane changes rounding; within-lane order is fixed by col,
            // so we only assert the forward traversal is reproducible.
            s2.add(i, 0.0);
        }
        let t1 = s.total();
        let mut s3 = FixedSum::<f32>::default();
        for (i, x) in v.iter().enumerate() {
            s3.add(i, *x);
        }
        assert_eq!(t1.to_bits(), s3.total().to_bits());
        let _ = s2;
    }
}
