# Limitations (v0.1)

**Physics**
- First-order finite volume. Fronts are numerically diffusive, and there is no MUSCL or second-order option yet.
- No culverts, bridges, subgrid obstructions, or buildings as porosity. Buildings are either ground cells or solid walls (`building_mode`).
- Rainfall is spatially uniform. Infiltration is a constant capacity, with no Green–Ampt.
- Transmissive boundaries are zero-gradient and partially reflective for subcritical or oblique flow.

**Grids and fidelity**
- Grids must be uniform, north-up and projected. There is no reprojection: do it beforehand with `gdalwarp`.
- Coarse fidelity levels keep thin barriers as face crests, but cannot rank primitives narrower than about 2–3 coarse cells. Such primitives are flagged as `unresolved` in `search.jsonl`.

**Results and interpretation**
- Peak-depth differences inside a cut include the water stored in the cut itself. Guard-area worsening is the constraint that protects third parties.
- The optimizer finds model-optimal designs on the supplied DEM. Results depend on DEM accuracy, roughness and the design storm. They are not engineering approval, and they must be validated at a finer resolution and against alternative storms (spec §9.2).

**Implementation status**
- The GPU backend is `f32` only and supports level-1 (Signed-face) runs with wall and transmissive boundaries. Coarse levels, stage/inflow boundaries, local-inertial physics, subdomain replay and robust ensembles run on the CPU. GPU candidates are not early-aborted, and GPU traces match the CPU only within tolerance (D3).
- Single-run thread scaling flattens beyond about 4 threads on a 12-core (8P + 4E) laptop under background load (spec Appendix B.3). It has not been measured on an idle workstation.
- The robust objective scales rain and inflow uniformly per member. There are no spatially varying storms or roughness perturbations yet.
