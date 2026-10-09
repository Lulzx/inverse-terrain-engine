//! 64-byte aligned buffers (spec §4.1). `unsafe` in the solver lives here and in `pool`.

use itr_core::Real;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u8; 64]);

/// A `[T]` whose first element is 64-byte aligned. `T` is `f32` or `f64`.
pub struct AlignedBuf<T: Real> {
    lines: Vec<Line>,
    len: usize,
    _t: core::marker::PhantomData<T>,
}

impl<T: Real> AlignedBuf<T> {
    pub fn new(len: usize, fill: T) -> Self {
        let bytes = len * core::mem::size_of::<T>();
        let nlines = bytes.div_ceil(64).max(1);
        let mut b = Self { lines: vec![Line([0; 64]); nlines], len, _t: core::marker::PhantomData };
        b.as_mut_slice().fill(fill);
        b
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `lines` holds at least `len * size_of::<T>()` initialized bytes, is
        // 64-byte aligned (≥ align_of::<T>()), and T is a plain float with no invalid
        // bit patterns.
        unsafe { core::slice::from_raw_parts(self.lines.as_ptr() as *const T, self.len) }
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above; `&mut self` guarantees exclusive access.
        unsafe { core::slice::from_raw_parts_mut(self.lines.as_mut_ptr() as *mut T, self.len) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Real> Clone for AlignedBuf<T> {
    fn clone(&self) -> Self {
        Self { lines: self.lines.clone(), len: self.len, _t: core::marker::PhantomData }
    }
}

impl<T: Real> core::ops::Deref for AlignedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Real> core::ops::DerefMut for AlignedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Padded row-major layout with a 1-cell ghost halo. The first interior cell of every
/// row (padded column 1) sits on a 64-byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub nx: usize,
    pub ny: usize,
    /// Elements before padded column 0 in each row.
    pub lp: usize,
    pub stride: usize,
}

impl Layout {
    pub fn new<T: Real>(nx: usize, ny: usize) -> Self {
        let lanes = T::LANES;
        let lp = lanes - 1;
        let stride = (lp + nx + 2).div_ceil(lanes) * lanes;
        Self { nx, ny, lp, stride }
    }
    /// Index of padded cell (i ∈ 0..nx+2, j ∈ 0..ny+2).
    #[inline(always)]
    pub fn at(&self, i: usize, j: usize) -> usize {
        j * self.stride + self.lp + i
    }
    #[inline(always)]
    pub fn row(&self, j: usize) -> core::ops::Range<usize> {
        let s = j * self.stride + self.lp;
        s..s + self.nx + 2
    }
    pub fn len(&self) -> usize {
        (self.ny + 2) * self.stride
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_interior_cell_aligned() {
        let l = Layout::new::<f32>(37, 5);
        let b = AlignedBuf::<f32>::new(l.len(), 0.0);
        for j in 0..7 {
            let p = &b[l.at(1, j)] as *const f32 as usize;
            assert_eq!(p % 64, 0);
        }
        let l = Layout::new::<f64>(37, 5);
        let b = AlignedBuf::<f64>::new(l.len(), 0.0);
        assert_eq!((&b[l.at(1, 3)] as *const f64 as usize) % 64, 0);
    }
}
