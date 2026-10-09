# Validation

Run `itr validate [--precision f32|f64|both] [--json]`. Each case uses the production fused kernel.

| Case | Metric | Threshold (f64 / f32) |
|---|---|---|
| Lake at rest over a variable bed with a step | max \|Δη\| + \|q\| | 1e-12 / 1e-4 |
| Rain on a closed flat plain | relative mass residual | 1e-12 / 1e-6 |
| Ritter dam break (dry bed) | relative L1 depth error vs analytic | 3e-2 |
| Wetting/drying over a step | mass residual; min h ≥ 0 | 1e-12 / 1e-6 |
| Rain on a tilted plane (kinematic limit) | \|Q_out/Q_rain − 1\| at steady state | 2e-2 |
| Gauge invariance (terrain and stage +1000 m) | cells differing, bitwise | 0 |
| Walls under rain | wall depth exactly 0; mass residual | 1e-12 / 1e-6 |
| Inflow boundary ledger | face flux vs storage residual | 1e-12 / 1e-6 |

## Implementation equivalence

`cargo test -p itr-hydro --release` checks the following:

- The fused kernel equals the scalar oracle bitwise.
- Results are bitwise identical with dry skipping on and off, across 1–7 threads with strip heights 1–43, and on edited terrain.
- Warm start equals cold start bitwise.
- Recording outputs does not change the trajectory.
- The mirror metamorphic test passes.
- Coarse levels keep a lake at rest.
- Steady state makes zero heap allocations (checked with a counting allocator), on the calling thread and across the persistent grid workers (`tests/pool_alloc.rs`).
- Baseline-locked Δt with a zero edit equals the baseline bitwise, and a violated hard CFL falls back adaptively (`tests/replay.rs`).
- Subdomain replay with a zero edit equals the baseline inside the ROI bitwise. A berm at the ROI ring is flagged `ReplayInvalid`.
- Moving a run from 1 → 4 → 2 threads mid-run (tail re-partitioning) is bitwise identical.
- Local-inertial kernel: lake at rest, rain mass, tilted-plane outflow, and peak depths within 15% L1 of HLL (`tests/inertial.rs`).
- Property tests (`proptest`): lake at rest on random terrains with wet/dry fronts and walls; `h ≥ 0` and ledger closure under random rain, edits and thread counts, for both kernels (`tests/properties.rs`).

## Optimization

`cargo test -p itr-opt --release` checks the following:

- The trace is identical for 1 and 3 workers.
- A resumed search with a partially filled cache reproduces the uninterrupted trace.
- Zero intervention gives a bitwise-identical baseline.
- Every guard-aborted candidate, re-run to completion, is infeasible.
- The robust objective equals `mean + β·CVaR_α` over the members, bitwise.
- `f32` and `f64` rank 16 random designs identically (Kendall τ, best design, max |ΔJ|).
- Exact Shapley values of the ablation sum to `J(none) − J(all)`.

## GPU

`cargo test -p itr-gpu --release` (skips without an adapter) checks the following:

- Lake at rest and rain volume.
- Tilted-plane outflow matches the CPU.
- A batch equals single-candidate runs bitwise (D2).
- Unsupported modes are rejected.

`itr bench --backend gpu` reports the max |Δ peak depth| against CPU `f32` (D3).

`examples/infeasible-problem` must exit with code 2.
