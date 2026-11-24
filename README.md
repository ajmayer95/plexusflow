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
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,figures]"
```

`gui` installs the viewer (napari, Qt) and `figures` what the figure scripts need; leave out either if you do not need
it. Other extras: `vectorize` ([net3](https://github.com/ajmayer95/net3), mask → graph), `bench` (Stokes benchmark),
`dev` (pytest, ruff). The exact environment in which the tests and figures were checked is pinned in
`requirements-lock.txt` (`pip install -r requirements-lock.txt && pip install -e . --no-deps`).

In a new terminal, activate the environment again with `source .venv/bin/activate` from the `plexusflow` folder.

## Getting the data

The data for the 21-somite embryo will be archived on Zenodo. Until then it is on this repository's
[Releases page](https://github.com/ajmayer95/plexusflow/releases) as **data-v4**, in two archives that unpack into
the same folder:

- `PlexusFlow21S_release_v4.core.tar.gz`: the measured vessel graph, results, mosaic, masks, per-tile projections and
  the other figure inputs;
- `PlexusFlow21S_release_v4.extras.tar.gz`: one short, downsampled video per tile, used for video playback in the
  viewer and for Figs 1 and 4.

Download both from the Releases page, or with the [GitHub CLI](https://cli.github.com):

```bash
mkdir -p $HOME/plexusflow_data
gh release download data-v4 -R ajmayer95/plexusflow -D $HOME/plexusflow_data
tar -xzf $HOME/plexusflow_data/PlexusFlow21S_release_v4.core.tar.gz -C $HOME/plexusflow_data
tar -xzf $HOME/plexusflow_data/PlexusFlow21S_release_v4.extras.tar.gz -C $HOME/plexusflow_data
export PLEXUSFLOW_STAGE=$HOME/plexusflow_data/PlexusFlow21S_release_v4
```

There is no default data location: every command stops with a clear message if `PLEXUSFLOW_STAGE` is unset. The
folder can be read-only.

## Exploring the data in the viewer

```bash
plexusflow view $PLEXUSFLOW_STAGE
```

This opens the mosaic with every vessel coloured by a chosen quantity: mean flow, pulsatile amplitude, phase lag
behind the vitelline veins, or relative pulsatile power. Three sources can be shown, alone or side by side on a common
colour scale:

- **Measured**: the per-vessel optical-flow measurements before reconciliation;
- **Reconciled**: the mass-conserving mean flow and the synchronised pulsatile field;
- **Model**: the RC network driven by the reconciled boundary flows, with a slider for the wall distensibility
  (starting at the fitted value).

Clicking a vessel shows its waveform over one cardiac cycle and the raw recordings from each tile it appears in. The
release already contains all results, so the viewer needs nothing else to be run first, and it never changes the
data.

## Reproducing the paper

With the data downloaded and `PLEXUSFLOW_STAGE` set as above, choose where derived results and figures go, then run
everything (about 4 minutes on a laptop):

```bash
export PLEXUSFLOW_RESULTS=$HOME/plexusflow_out/results
export PLEXUSFLOW_FIGURES=$HOME/plexusflow_out/figures
bash scripts/reproduce_paper.sh
```

Figures 1–7 end up in `$PLEXUSFLOW_FIGURES/latest/`. `bash scripts/reproduce_paper.sh --figures` redraws the figures
from existing results. If the two variables are unset, results and figures are written inside the data folder, which then has to be
writable.

| Step | Script | Output (in `$PLEXUSFLOW_RESULTS`) |
|---|---|---|
| mean flow (conservation, equal left/right inflow) | `scripts/dc_reconcile.py` | `dc_reconciled_v3.pkl` |
| tile clocks and pulsatile flow | `scripts/pulse_field.py` | `ac_field_v9.pkl` |
| distensibility fit | `scripts/mech/ac_estimate.py` | `ac_estimate_v11.json` |
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

### Data written by earlier versions

Graphs, sidecars and releases written before 2026-11-24 use older names for the per-vessel flux estimator (parabolic
profile fits per cross-section).  They are read as they are: every loader passes stored data through
`plexusflow.compat`, the only module that knows the old names, and a new batch writes only the current ones
(`Q_profile`, `quality.profile_ok`, `batch_profile_inputs/`, …).  To rewrite a stage folder in the current names
(optional; the graph is backed up first, run with system python3 / numpy < 2):

```bash
python3 -m plexusflow.compat "$PLEXUSFLOW_STAGE"
```

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
