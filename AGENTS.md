# Guide for AI coding agents (and new contributors)

Read this first, then [README.md](README.md) (install, reproduce), [docs/method.md](docs/method.md) (the
method and its numbers) and [docs/pipeline.md](docs/pipeline.md) (end-to-end pipeline).

## What this repository is

- **`plexusflow`:** reconstructs blood flow in the embryonic yolk-sac vascular plexus from tiled 250 fps bright-field movies:
  vessel graph → per-vessel optical-flow flux → mass-conserving mean flow (DC) → tile-synchronised pulsatile (AC) field.
- **`plexusmech`:** fits a distributed RC wall-mechanics model to that field (distensibility D).
- **Paper:** PLOS Comp Biol methods paper, 21-somite embryo (stage "21S").

## Getting the code and data

```bash
git clone https://github.com/ajmayer95/plexusflow.git && cd plexusflow
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[figures]"            # + [gui] for the napari viewer
```

**Data are not in git.** A stage is a folder, e.g. `PlexusFlow21S_release_v3/`, shared separately by the authors (Zenodo on
publication). Extract `*.core.tar.gz`, and optionally `*.extras.tar.gz` (tile videos for viewer playback, Kloosterman
sidecars), into the same folder. Then:

```bash
export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release_v3     # required — no default; scripts stop if unset
export PLEXUSFLOW_RESULTS=$HOME/pf_out/results                 # where derived results go (default <stage>/results)
export PLEXUSFLOW_FIGURES=$HOME/pf_out/figures                 # where figures go (default <stage>/figures)
```

Point RESULTS / FIGURES at your own folders so the frozen release is never modified. A release contains:
- `graph/graph_vessels_piv.gpickle`: the measured graph (Q(t) traces re-attached from `graph/export/traces.npz` by `plexusflow.paths.load_graph`);
- `graph/export/*.csv`: portable tables, see `SCHEMA.md`;
- `fuse/` (mosaic), `stitch/` (tile placement), `results/` (final outputs), `MANIFEST.json` (checksums, code commit).

## Common tasks

| task | command |
|---|---|
| everything for the paper | `bash scripts/reproduce_paper.sh` |
| analysis chain for one stage | `bash scripts/run_stage.sh "$PLEXUSFLOW_STAGE"` (`--no-ac`) |
| mean-flow reconstruction only | `python scripts/dc_reconcile.py` |
| figure N | `python scripts/make_figureN.py` (see [FIGURES.md](FIGURES.md)) |
| read-only viewer | `python -m plexusflow.cli view "$PLEXUSFLOW_STAGE"` (needs `[gui]`) |
| tests | `pytest -q` |

## Programmatic use

```python
import plexusflow.paths as P                                  # resolves PLEXUSFLOW_STAGE / RESULTS
from plexusflow.reconcile import measurement_table, fit_reliability, carved_domain, reconcile_dc
G = P.load_graph()                                            # networkx graph, edges carry measurements_piv (per tile)
M = measurement_table(G)                                      # {(tile, (u, v)): row}: Q (signed u→v), cv, align, …
model, junctions = fit_reliability(G, M)                      # reliability model: used by the earlier estimators only
D = reconcile_dc(G, model, M, dom=carved_domain(G))           # D["Qp"] reconciled flux per carved edge, D["fb"] boundary fluxes
```

Edge orientation: every edge has a tail → head orientation and positive flux runs tail → head. A measurement's
`flow_from` / `flow_to` give its own direction.

## Current method decisions (don't change without discussion)

- **Flux per recording:** per-frame Kloosterman fit with geometry frozen from the time-mean fit. A failed Kloosterman fit means *unmeasured*; there is no fallback estimator.
- **Fit per recording:** mean + 2 harmonics at the tile heart rate (`reconcile.dc.K_FIT = 2`).
- **One uncertainty per recording:** σ² = s²/M from the residual of that fit; every weight is 1/σ² (mean flow, tile clocks, harmonic average). The reliability model c is not used.
- **Mean flow:** 1/σ² average of the fitted means, projected onto junction conservation with total left = total right arterial inflow (`reconcile.dc.SYMMETRIC = True`).
- **Tile clocks:** `plexusflow.ac.speed` + `ac.pulse.tile_clocks`: speed pulse of the velocity pooled over the common piece of a vessel seen in two tiles (from the raw movies, no profile fit, no flow direction); recordings whose velocity reverses within the beat are left out; weighted circular least squares, solved jointly; tiles with clock SD > 30° are not synchronised.
- **Pulsatile flow:** `plexusflow.ac.pulse`: flux harmonics of the recordings that agree with the reconciled flow direction, averaged on the common clock and rescaled to the reconciled mean; recordings of the opposite direction are never sign-flipped; vessels with both directions are flagged.
- **Distensibility fit:** boundary forcing as measured, clocks fixed, no group corrections, one common variance per loss term (`scripts/mech/ac_estimate.py --method measured`).
- **Result files:** `dc_reconciled_v3.pkl`, `speed_pairs_v1.pkl`, `ac_field_v9.pkl`, `ac_estimate_v10.json` (`ac_field_v8` / `ac_estimate_v9`: the earlier method, sensitivity). Earlier estimators stay selectable and write other names.

## Conventions and pitfalls

- **Package boundary:** `plexusmech` may import `plexusflow`, never the reverse (`tests/test_package_boundary.py`).
- **numpy version:** stage graphs are pickled with **numpy 1.x**. Write any graph pickle with numpy < 2, or other environments can't read it. Reading with numpy 2 is fine.
- **Where scripts go:** analysis drivers and figure scripts go in `scripts/`. Exploratory work goes in `sandbox/`, which is gitignored and never committed. Long runs: tee to `run_<name>.log`.
- **Never commit:** data, results or figures.
- **Units:** flux in nL/s, radii and lengths in pixels of the mosaic (1.7 µm/px unless the stage config says otherwise), heart rate in Hz, D in Pa⁻¹.
- **Viewer attributes:** the viewer attaches in-memory attributes prefixed `_rec_` / `_h_*_rec`. They are stripped on save; never persist them.
- **Topology edits:** use the workbench (`python -m plexusflow.cli workbench …`). It keeps a reversible patch log under `graph/patches/`. Re-measure and rerun `run_stage.sh` afterwards.

## Where things are

| path | contents |
|---|---|
| `src/plexusflow/analyze/` | optical flow, Kloosterman flux (`kloosterman.py`), Q(t) (`profile.py`, `mosaic_piv.py`), motion diagnostics |
| `src/plexusflow/reconcile/` | measurement table and uncertainty model (`reliability.py`), reconstruction (`dc.py`, `network.py`), gains (`gain.py`), QC |
| `src/plexusflow/ac/` | pulsatile field |
| `src/plexusmech/` | RC model and distensibility fit |
| `src/plexusflow/viewer/` | napari viewer: `app.py`; edge flux card in `edge_card.py` |
| `scripts/` | drivers, figures, `make_release.py` (builds a data release from a stage) |
| `docs/` | methods notes (`method.md`, `pipeline.md`, `synthetic_validation.md`) |
