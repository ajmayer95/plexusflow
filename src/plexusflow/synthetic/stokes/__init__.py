"""Known-truth benchmark of the per-vessel flux measurement (docs/synthetic_validation.md).

Exactly conserving Stokes flow in real junction and controlled vessel geometries → tracers advected on the exact field →
hybrid videos on real tile backgrounds with that camera's noise → the unchanged production measurement → compare.

  geometry   build junction / vessel meshes and Stokes solutions (gmsh, scikit-fem; bundle building only)
  tracers    exact P2 field evaluation, pseudo-time advection, exposure-blurred tracer renderer
  render     hybrid video model (background, tracer scale, jitter, camera noise)
  measure    production measurement wrappers (flux, quality, H1/DC vs truth)
  bundle     the benchmark bundle (backgrounds + saved Stokes solutions) — runs need nothing else
  cases      junction_case, vessel_case

Drivers: scripts/benchmark/{make_bundle,junctions,vessels,frame_spacing}.py.  Needs the `bench` extra
(scikit-fem, numba; gmsh + meshio only to rebuild geometry).
"""
