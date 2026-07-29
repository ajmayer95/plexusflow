# plexusflow

**Reproducible whole-organ vascular reconstruction from spatially tiled flow microscopy.**

`plexusflow` converts overlapping microscopy tiles into an analysis-ready
whole-organ vascular graph with calibrated, edge-resolved flow
measurements and documented uncertainty. It is designed for planar
vascular / tubular networks (e.g. embryonic yolk-sac vasculature) imaged
in overlapping fields of view.

## Status

Pre-alpha. Under active development. APIs unstable.

## Install

```bash
pip install -e ".[dev]"
```

Optional backends and interfaces:

```bash
pip install -e ".[gui]"       # napari-based interactive editor
pip install -e ".[sam2]"      # SAM2 segmentation backend
pip install -e ".[ilastik]"   # ilastik output loader
```

## Pipeline

```text
raw tiles
   ↓
ingest ─────────── tile discovery + metadata
   ↓
register ───────── pose-graph global alignment
   ↓
segment ────────── pluggable backends (threshold / ilastik / SAM2)
   ↓
fuse ───────────── probability fusion in global coords
   ↓
vectorize ──────── centerline extraction + graph build
   ↓
edit ───────────── mask / graph correction (napari)
   ↓
measure ────────── local flow estimation per edge   [Phase 2]
   ↓
harmonize ──────── cross-tile calibration           [Phase 3]
   ↓
qc ─────────────── metrics + HTML report
```

Every stage is idempotent, resumable, and callable from both Python and
the CLI.

```bash
plexusflow register  project.yaml
plexusflow segment   project.yaml --backend ilastik
plexusflow fuse      project.yaml
plexusflow vectorize project.yaml
plexusflow qc        project.yaml
```

```python
from plexusflow import Project
proj = Project.load("project.yaml")
proj.register().segment().fuse().vectorize()
proj.qc().write_report("qc/report.html")
```

## Authors

- **Alex Mayer** — Katifori Lab, Center for Computational Biology,
  Flatiron Institute.

## License

MIT — see [LICENSE](LICENSE).
