//! Inverse Terrain Engine core: grid conventions, scenario schema, model contracts,
//! earthwork design space, and objective definitions. No I/O, no threads, no GPU.
#![forbid(unsafe_code)]

// `!(a > b)` is deliberate NaN-aware logic; index loops mirror the stencil maths.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]
pub mod design;
pub mod hash;
pub mod model;
pub mod objective;
pub mod raster;
pub mod real;
pub mod scenario;

pub use real::{max_slice, FixedSum, Real};

/// Engine model version; part of every cache key and manifest.
pub const MODEL_VERSION: &str = concat!("itr-", env!("CARGO_PKG_VERSION"), "-fv1_hll_hr");
