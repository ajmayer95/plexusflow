# Roadmap

Phase 1 — geometry: **image → global mask → editable graph**.
Phase 2 — measurement: local flow → edge assignment.
Phase 3 — harmonization: cross-tile calibration + uncertainty.
Phase 4 — application: reference simulator + validation.

## Phase 1 milestones

- [x] 1a. Walking skeleton: repo bootstrap, CI green, `pip install -e .` works.
- [ ] 1b. Registration: pose-graph global stitch + QC.
- [ ] 1c. Segmentation backend framework (threshold / ilastik / SAM).
- [ ] 1d. Probability fusion in global coords + uncertainty map.
- [ ] 1e. Vectorization + graph with per-edge tile provenance.

Success criterion for Phase 1: run `plexusflow demo` on the
`examples/4tile_demo/` fixture end-to-end and produce a `graph.graphml`
and `qc/report.html`, reproducibly, on every supported Python.
