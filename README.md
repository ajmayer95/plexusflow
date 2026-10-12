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

## Quickstart for collaborators (frozen release)

You need: access to this repository, a copy of the data release (`PlexusFlow21S_release_v1/`, ~375 MB — shared
separately), and Python ≥ 3.10.

```bash
git clone https://github.com/ajmayer95/plexusflow.git && cd plexusflow
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                       # add ".[gui]" for the napari viewer

export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release_v1   # the release (treated as read-only)
export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results       # where your outputs go
export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures

python examples/quickstart.py          # load graph → reliability fit → DC reconciliation → 2 plots (~1 min)
python scripts/dc_reconcile.py         # the full DC pipeline (writes reliability_model_v1.json, dc_reconciled_v2.pkl)
python scripts/make_dc_emergent_figure_v2.py      # DC figures (after dc_reconcile)
python scripts/make_figure8alt_v3.py
python scripts/ac_field.py             # AC harmonic field (after dc_reconcile)
python scripts/ac_estimate.py          # joint (D, η) estimate — slow (tens of minutes)
python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"   # read-only viewer (needs the gui extra)
```

Working without Python: `graph/export/` in the release holds the graph and every measurement as CSV + npz
(`SCHEMA.md` explains the columns); `results/` holds the reconciled flows and AC field as CSV.
The release's `results/` were produced with this code; re-running reproduces them exactly in the environment recorded in
its `MANIFEST.json`, and to ~0.3 % with other numpy/scipy versions.

## Repository layout (2026-10)

```text
src/plexusflow/
  analyze/       optical flow + Kloosterman per-vessel batch (per-measurement `quality` record)
  reconcile/     DC: reliability scale, carved network, reliability-weighted IRLS projection, passive flow
  ac/            AC: Option-A harmonic field, two-port network model, likelihood, group-corrected (D, η) estimate
  ...            graph I/O, viewer, CLI and the upstream stages
scripts/         pipeline drivers, run in order:
  dc_reconcile.py                  → results/reliability_model_v1.json, results/dc_reconciled_v2.pkl
  ac_field.py                      → results/ac_field_v2.pkl
  ac_estimate.py                   → results/ac_group_bc_estimate.pkl, figures/ac/
  make_dc_emergent_figure_v2.py, make_figure8alt_v3.py     DC figures
  make_release.py                  → frozen, self-contained data release of a stage
examples/        quickstart.py — graph → reconciliation → plots
docs/            methods notes (docs/dc_method.md)
```

Only pipeline code is committed.  Diagnostics, validation runs and exploration live in `sandbox/`
(git-ignored; on the main Mac a link to a Drive folder so both machines see it).

**Data is not in git.** The canonical measured graph (`graph/graph_vessels_piv.gpickle`), Kloosterman
sidecars, figures and derived results live in the stage folder (`PlexusFlow21S/`, synced via Google Drive);
raw videos stay on the acquisition machine. Point the code at the stage folder once per machine:

```bash
export PLEXUSFLOW_STAGE="/path/to/PlexusFlow21S"     # add to ~/.zshrc; default = the Drive path on the main Mac
```

`plexusflow.paths` provides `STAGE`, `RESULTS` (= `<stage>/results`, or `$PLEXUSFLOW_RESULTS`), `GRAPH`, `FIGURES`.
Derived results are regenerated by code and shared between machines through the stage folder.

## Workflow

Two machines: code syncs through GitHub, data through the stage folder.
`git pull` when you start, commit as you go, `git push` when you stop.

- `main` holds only finished, working code. Work happens on branches (`git switch -c <topic>`);
  commit freely there, merge to `main` by pull request when satisfied.
- Diagnostics and exploration go in `sandbox/` (git-ignored), never in a commit.  When the pipeline needs
  something from a diagnostic, move that function into `src/plexusflow/`.
- Data versions are published separately with a checksum; results record the graph version they came from.
