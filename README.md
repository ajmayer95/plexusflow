# plexusflow

**Whole-organ blood-flow reconstruction from tiled high-speed microscopy of a planar vascular network.**

`plexusflow` turns overlapping 250 fps bright-field movies of the embryonic yolk-sac plexus into a vessel graph with a
measured, mass-conserving mean flow and a synchronised pulsatile (harmonic) field on every vessel. `plexusmech` fits a
distributed RC network model of the wall mechanics to that field. This repository contains both packages, the
analysis drivers and the scripts that produce every figure of the paper.

> **Paper:** A. Mayer *et al.*, *title*, PLOS Computational Biology (in review). <!-- TODO: title, DOI on acceptance -->
> **Code archive:** [doi:10.5281/zenodo.TODO](https://doi.org/10.5281/zenodo.TODO) ·
> **Data:** [doi:10.5281/zenodo.TODO](https://doi.org/10.5281/zenodo.TODO)

## What is in the repository

| Package / folder | Contents |
|---|---|
| `src/plexusflow/` | reconstruction: tile preprocessing and stitching, mosaic fusion, mask → graph (`mve`, `vectorize`), per-vessel optical-flow flux (`analyze`), mean-flow reliability model and conservative reconciliation (`reconcile`), tile-synchronised harmonic field (`ac`), known-truth Stokes benchmark (`synthetic`), data-release loader (`release`), napari viewer (`viewer`) |
| `src/plexusmech/` | modelling: two-port RC / transmission-line network (`ac.model`, `tl`), composite likelihood and profiled fit of wall distensibility (`ac.profile`) |
| `scripts/` | analysis drivers (`dc_reconcile.py`, `pulse_field.py`, `mech/ac_estimate.py`; `run_stage.sh` runs them for any stage), figure scripts (`make_figure*.py`, see [FIGURES.md](FIGURES.md)), benchmark drivers (`benchmark/`), `reproduce_paper.sh` |
| `docs/` | methods notes: [pipeline](docs/pipeline.md), **[the method](docs/method.md)**, [synthetic validation](docs/synthetic_validation.md) |
| `AGENTS.md` | guide for AI coding agents and new contributors (setup, data, commands, conventions) |
| `tests/` | unit and regression tests (`pytest`) |

`plexusmech` imports `plexusflow`, never the reverse (enforced by `tests/test_package_boundary.py`).

## Install

Python ≥ 3.10.

```bash
git clone https://github.com/ajmayer95/plexusflow.git
cd plexusflow
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[figures]"                 # library + everything the figure scripts need
```

The exact environment in which the tests and all figures were verified is pinned in `requirements-lock.txt`
(`pip install -r requirements-lock.txt && pip install -e . --no-deps`). Optional extras: `gui` (napari viewer and
editors), `vectorize` ([net3](https://github.com/ajmayer95/net3), mask → graph), `bench` (Stokes benchmark), `dev`
(pytest, ruff).

## Reproducing the paper

1. Download and extract the data release (21-somite embryo) from Zenodo (link above). It contains the measured vessel
   graph, the stitched mosaic and masks, per-tile projections and the inputs of every figure.
2. Point the code at it. There is no default location: every script stops with a clear message if
   `PLEXUSFLOW_STAGE` is unset.

   ```bash
   export PLEXUSFLOW_STAGE=/path/to/PlexusFlow21S_release       # read-only is fine
   export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results       # derived results (default <stage>/results)
   export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures       # figures (default <stage>/figures)
   ```
3. Run everything (about 4 minutes on a laptop):

   ```bash
   bash scripts/reproduce_paper.sh             # results, then Figs 1–7 → $PLEXUSFLOW_FIGURES/latest/
   bash scripts/reproduce_paper.sh --figures   # figures only, from existing results
   ```

| Step | Script | Output (in `$PLEXUSFLOW_RESULTS`) |
|---|---|---|
| mean flow (conservation, equal left/right inflow) | `scripts/dc_reconcile.py` | `dc_reconciled_v3.pkl` |
| tile clocks and pulsatile flow | `scripts/pulse_field.py` | `ac_field_v9.pkl` |
| distensibility fit | `scripts/mech/ac_estimate.py` | `ac_estimate_v10.json` |
| Figures 1–7 | `scripts/make_figure*.py` ([FIGURES.md](FIGURES.md)) | `$PLEXUSFLOW_FIGURES/figN/` |

Script defaults are the settings used in the paper; the method is described in [docs/method.md](docs/method.md).

## Running the pipeline on new data

The upstream steps work on a *stage folder* (raw-tile movies in, graph and measurements out), each a `plexusflow`
sub-command with `--help`:

```bash
plexusflow init        STAGE        # folder layout + plexusflow.yaml
plexusflow preprocess  STAGE        # per-tile projections from the raw movies
plexusflow stitch      STAGE        # interactive tile placement (PerTileFlow GUI, see docs/pipeline.md)
plexusflow fuse        STAGE        # feather-blended mosaic
plexusflow mve         STAGE        # mask → vectorise → edit (napari, needs [gui,vectorize])
plexusflow condense    STAGE        # one edge per vessel
plexusflow analyze     STAGE        # optical flow → Q(t) → harmonics per vessel and tile
plexusflow workbench   STAGE        # mark aorta/vein anchors and tag open / topology-uncertain regions
bash scripts/run_stage.sh STAGE     # mean flow, pulsatile field, distensibility fit
plexusflow view        STAGE        # read-only viewer: measured / reconciled flow, RC-network model with a distensibility slider, time series
plexusflow info        STAGE        # which artefacts exist
```

See [docs/pipeline.md](docs/pipeline.md) for every step, the per-stage settings in `plexusflow.yaml`, and the coordinate
frames and outputs.

## Tests

```bash
pip install -e ".[dev]"
pytest                    # unit + regression tests (DC network, reliability fit, AC profile, package boundary, benchmark)
ruff check src tests scripts examples
```

`tests/test_release.py` additionally checks a data release when `PLEXUSFLOW_TEST_RELEASE` points at one.

## Citation

Please cite the paper and the archived code version ([CITATION.cff](CITATION.cff)).

## Authors and acknowledgements

Alex Mayer — Katifori Lab, Center for Computational Biology, Flatiron Institute. Imaging: Siavash Ghaffari, Elizabeth
A. V. Jones lab. <!-- TODO: confirm acknowledgement wording and affiliations -->

## License

MIT — see [LICENSE](LICENSE).
