//! Optimizer layer (spec §6): pinned RNG, CMA-ES and baselines, candidate evaluator with
//! exact early abort and caching, multi-fidelity search driver. Depends on the solver
//! only through `itr-hydro`'s public run API.
#![forbid(unsafe_code)]
// Index loops mirror the linear-algebra / sequence maths.
#![allow(clippy::needless_range_loop)]

pub mod ablation;
pub mod cmaes;
pub mod corridor;
pub mod driver;
pub mod evaluator;
pub mod problem;
pub mod rng;
pub mod search;
pub mod sobol;

pub use driver::{search, NullSink, SearchEntry, SearchResult, Sink};
pub use evaluator::{EvalRecord, Evaluator, Status};
pub use problem::{Problem, ProblemInputs};
