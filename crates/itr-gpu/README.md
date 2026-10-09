# itr-gpu

Optional wgpu/WGSL GPU backend for the inverse terrain engine (spec 7.8). A workspace member
outside `default-members`, so plain `cargo build` does not compile wgpu.

    cargo test --release -p itr-gpu
    cargo run --release -p itr-gpu --example gpubench
    cargo build --release -p itr-cli --features gpu   # itr bench --backend gpu; solver.backend = "gpu"

## Status

Working, tested on Apple M4 Pro (Metal). API: `GpuSolver::new()`, `adapter_info()`,
`run_batch(&SolverParams, &[&PreparedTerrain<f32>], &MonitorSet) -> Vec<GpuRun>`.

## What is implemented

- f32 only; numerics are a line-by-line port of `itr-hydro/src/numerics.rs`: depth-only
  hydrostatic reconstruction from signed face jumps, HLL with the same wave speeds and
  dry-dry zero, desingularized velocity, rain, infiltration `min(cap*dt, h)`, point-implicit
  Manning friction with the multiplication-only `rcbrt` (same f32 seed and 3 Newton steps),
  positivity clamp.
- Baseline variant of 7.8: per step one `update` dispatch (16x16 tiles with halo in
  workgroup memory, ping-pong via two state halves selected by a per-candidate parity) and
  one 1-workgroup-per-candidate `finalize` dispatch. dt = cfl/smax from an `atomicMax` of
  `bitcast<u32>(abs(s))`, capped at `dt_max`, clipped to the next sync instant (same rule as
  the CPU loop). Non-finite values set a per-candidate `bad` flag instead.
- Time is a double-single f32 pair; the rain integral over `[t, t+dt]` is computed exactly
  on the device from the piecewise-linear table (relative coordinates).
- 64 steps (configurable `steps_per_submit`) per command buffer; the host reads only the
  control buffer between submits and compacts the active-candidate list, so finished or
  bad candidates leave the dispatch.
- Candidate batching along dispatch z, per-candidate face-jump overlays.
- Monitored running maxima of depth via `atomicMax` on the f32 bit pattern.

## Limitations / deviations from 7.8

- `FaceMode::Signed` only (level 1); Crest mode returns `Err`.
- Boundaries: Wall and Transmissive. Stage and Inflow return `Err`.
- Two dispatches per step (no fused triple-buffered variant yet).
- No per-workgroup Kahan ledger, no ledger rows, no time-of-peak, no checkpoints or warm
  start, no abort observer. Volume is summed on the host in f64 from the final depth.
- Candidates must share one `Arc<TerrainBase<f32>>` (boundaries, wet mask, roughness come from it).
- Initial dt comes from a host pre-pass (f64), as on the CPU. `skip_dry` is ignored;
  new `SolverParams` fields (locked dt, hard CFL, inflow scale) are ignored.
- No dry-tile skipping. The whole state is re-allocated per `run_batch`.

## Measured (Apple M4 Pro, Metal, machine under load from other processes)

Tolerances (tests, observed):

| Test | Result |
|---|---|
| lake at rest, 64x48 variable bed, 1446 steps | max abs eta error 4.4e-7 m (limit 1e-4) |
| rain on closed walled plain | volume relative error 2.4e-8 (limit 1e-4) |
| D3 vs CPU `Solver<f32>`, tilted plane + rain + transmissive east | max monitor relative diff 1.6e-7 (limit 2%), identical step count 1718 |
| D2: batch of 3 edited candidates vs each alone | bitwise identical depth, monitors, steps, t_end |

Throughput (`examples/gpubench.rs`, useful cell-updates only, includes host polling):

| Case | G cell-updates/s |
|---|---|
| 512^2 x 1 candidate | 1.35 (CPU f32 single thread, same case: 0.10) |
| 16 x 128^2 batch | 0.66 |
| 64 x 128^2 batch | 1.15 |
| 1024^2 x 1 | 1.40 |

The small batch cases run only ~113 steps, so fixed costs (buffer setup, submit, readback)
are a large share; the numbers are far below the bandwidth bound and a fused variant and
fewer redundant face evaluations are the obvious next steps. GPU acceleration is not
claimed beyond these measurements (spec: not assumed until measured at matched accuracy).
