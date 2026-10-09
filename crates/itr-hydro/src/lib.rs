//! CPU shallow-water solver (spec §5, §7): `f32`/`f64` generic, scalar oracle and fused
//! in-place kernel sharing one set of numerics, mass ledger, sync points, checkpoints.

// `!(a > b)` is deliberate NaN-aware logic; index loops mirror the stencil maths.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]
pub mod aligned;
pub mod boundary;
pub mod inertial;
pub mod kernel;
pub mod numerics;
pub mod pool;
pub mod replay;
pub mod solver;
pub mod terrain;
pub mod validation;

pub use solver::{BaselineRecord, Checkpoint, KernelKind, Solver, SolverParams, Workspace};
pub use terrain::{FineTerrain, PreparedTerrain, TerrainBase};
