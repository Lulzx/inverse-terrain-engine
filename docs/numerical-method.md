# Numerical method

The full derivation and rationale are in spec §5 and §7. This page summarizes what the code does (`crates/itr-hydro/src/numerics.rs`).

## Equations

The 2D shallow-water equations in conservative form, with state `(h, qx, qy)`:

```
∂h/∂t  + ∂qx/∂x + ∂qy/∂y = r − i
∂qx/∂t + ∂(qx²/h + g h²/2)/∂x + ∂(qx qy/h)/∂y = −g h ∂z/∂x − g n² qx |q| / h^{7/3}
∂qy/∂t + ∂(qx qy/h)/∂x + ∂(qy²/h + g h²/2)/∂y = −g h ∂z/∂y − g n² qy |q| / h^{7/3}
```

## Discretization

**Spatial scheme**
- First-order finite volume on a uniform grid.
- Elevation is never stored next to depth. Each face carries the jump `Δz = z_R − z_L`, computed in `f64` and rounded once (spec §4.1).

**Face flux**
- Hydrostatic reconstruction (Audusse et al. 2004) in depth-only form: `h_L* = max(0, h_L − max(Δz, 0))` and `h_R* = max(0, h_R − max(−Δz, 0))`.
- The bed-slope correction `g/2 (h² − h*²)` is folded into the per-side normal flux, which makes the scheme well-balanced by construction.
- HLL flux with two-rarefaction wave-speed estimates. It is branch-free: selects, no `if`.
- Dry–dry faces return exactly zero, which makes dry skipping exact.
- Velocity is desingularized: `u = 2 h q / (h² + max(h², h_ε²))`.

**Time stepping**
- Explicit Euler with an unsplit 2D CFL condition: `Δt = C / max_cells((|u|+c)/Δx + (|v|+c)/Δy)`, with `C ≤ 0.5` and `Δt ≤ dt_max`.
- Δt is clipped only to land exactly on sync instants.

**Sources** (point-wise, after the flux update)
- A positivity fix clamps roundoff-negative depths to zero. The volume added is recorded in the ledger.
- Rain adds the exact integral of the hyetograph over `[t, t+Δt]`.
- Infiltration removes `min(capacity·Δt, h)`.
- Point-implicit Manning friction:

  ```
  q ← q / (1 + Δt g n² |q| h^{-7/3})
  ```

  Here `h^{-7/3} = r⁷` with `r = h^{-1/3}` from a bit-level seed and Newton steps, using multiplication only (no libm).

**Ledger**
- `V(t) − V₀ = rain + inflow − outflow − infiltration + positivity_fix + residual`.
- The residual is reported per sync interval.

## Determinism

The same results are guaranteed across thread counts, strip sizes, dry skipping, warm start and outputs. How:

- Kernel arithmetic is limited to `+ − × ÷ √` (no FMA contraction, no platform libm, no approximate reciprocal instructions).
- Every sum uses a fixed 16-accumulator layout.
- Strip-boundary faces are computed once in a pre-phase.
- The oracle and fused kernels share the numerics, so they agree bitwise.
