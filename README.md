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

## Quickstart for collaborators (frozen data releases)

Frozen data releases for 15, 21 and 27 somites are attached to the GitHub release
[`data-v2`](https://github.com/ajmayer95/plexusflow/releases/tag/data-v2) (repository access required). Each stage has a
**core** archive (graph, exports, mosaic, final results; 62–103 MB) and an optional **extras** archive (viewer video
playback, Kloosterman inputs); both extract into the same folder.  You need Python ≥ 3.10 and the GitHub CLI (`gh`).

Paste-safe commands (zsh does not accept `#` comments interactively, so none are inline):

```bash
gh release download data-v2 --repo ajmayer95/plexusflow --pattern 'PlexusFlow21S_release_v2.core.tar.gz'
tar -xzf PlexusFlow21S_release_v2.core.tar.gz
git clone https://github.com/ajmayer95/plexusflow.git
cd plexusflow
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export PLEXUSFLOW_STAGE="$PWD/../PlexusFlow21S_release_v2"
export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results
export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures
python examples/quickstart.py
```

| Command | What it does |
|---|---|
| `python examples/quickstart.py` | load graph → reliability fit → DC reconciliation → 2 plots |
| `python scripts/dc_reconcile.py` | full DC pipeline (writes `reliability_model_v1.json`, `dc_reconciled_v2.pkl`) |
| `python scripts/ac_field.py --ref-tile 21` | aligned waveform (AC) field — 27S: `--ref-tile 43`; 15S: `--ref-tile 15 --f0-band 10` |
| `python scripts/make_figure8alt_v3.py`, `python scripts/make_figure_ac_field.py` | DC figure (measured vs Poiseuille) and the measured AC field (after the two above) |
| `python scripts/mech/make_figure9alt_v3.py` | modelling: measured vs RC-network AC field (`plexusmech`) |
| `python scripts/mech/ac_estimate.py` | modelling: joint (D, η) estimate — slow (tens of minutes) |
| `pip install -e ".[gui]"` then `plexusflow view "$PLEXUSFLOW_STAGE"` | read-only viewer (napari; video playback needs the extras archive) |

From Python, a release is opened with `plexusflow.release.load_release(path)` (graph, DC and AC results, carved
network, tiles, checksum verification); modelling code should go through it rather than reading files directly.

Working without Python: `graph/export/` in a release holds the graph, every measurement and the per-tile heart rates as
CSV + npz (`SCHEMA.md` explains the columns); `results/` holds the reconciled flows and the waveform field as CSV.

Reproducibility: each release records the code commit it was built with (`MANIFEST.json`).  Re-running the scripts in a
fresh environment reproduces the reconciled flows of every vessel not flagged `unstable` to ≤ 0.6 % with no direction
changes (`results/dc_reconciled_edges.csv`: `Q_reconciled_sd`, `unstable` — weakly determined loop and near-zero flows,
0.03–0.12 % of vessels), and the tile phases of every tile with `phase_determined` waveforms exactly.  Waveform phases are
only comparable across tiles whose phase uncertainty is ≤ 30° (`results/summary.json`: `tile_time_gauge_sd_deg`).

## Repository layout (2026-10)

```text
src/plexusflow/      measurement → conservative flow field (paper 1)
  analyze/       optical flow + Kloosterman per-vessel batch (per-measurement `quality` record)
  reconcile/     DC: reliability scale, carve, carved network, reliability-weighted IRLS projection, passive flow
  ac/            AC: Option-A harmonic field (tile time gauge, junction residuals, phase-determined vessels)
  synthetic/     known-truth benchmark: Stokes flow in real junctions / vessels → hybrid videos → production measurement
  release.py     read-only loader for frozen data releases (schema-versioned)
  viewer/        napari viewer: `plexusflow view` (read-only) / `plexusflow workbench` (fresh OF, tagging, edits)
  legacy/        2026-09 tile_normalize / KCL path, kept to reproduce old figures; unused by pipeline and viewer
  ...            graph I/O, CLI and the upstream stages
src/plexusmech/      modelling on top of a reconstruction (paper 2) — imports plexusflow, never the reverse
  ac/            two-port network model, likelihood, group-corrected (D, η) estimate
  tl/            transmission-line simulator (`plexusmech simulate`); viewer/ backend for the Simulate tab
scripts/         pipeline drivers, run in order:
  dc_reconcile.py                  → results/reliability_model_v1.json, results/dc_reconciled_v2.pkl
  ac_field.py                      → results/ac_field_v2.pkl
  make_dc_emergent_figure_v2.py, make_figure8alt_v3.py, make_figure_ac_field.py     reconstruction figures
  make_release.py                  → frozen, self-contained data release of a stage
  benchmark/                       known-truth benchmark of the flux measurement (docs/synthetic_validation.md):
                                   make_bundle.py, junctions.py, vessels.py, frame_spacing.py
  mech/ac_estimate.py              → results/ac_group_bc_estimate.pkl, figures/ac/
  mech/make_figure9alt_v3.py       measured vs RC-network figure
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
