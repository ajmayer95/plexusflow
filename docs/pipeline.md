# plexusflow pipeline (raw videos → measured graph → reconstruction → viewer)

End-to-end walkthrough. Every stage reads/writes files under a single
per-embryo folder (a "stage root"), so `plexusflow <cmd> <STAGE_ROOT>`
works for any embryo without extra config.

## Stage folder layout

    PlexusFlow21S/
    ├── plexusflow.yaml               config overrides (optional)
    ├── tiles/
    │   ├── std_clahe/                vid{N}_std_clahe.tif  (per-tile std+CLAHE)
    │   ├── maxmed/                   vid{N}_maxmed.tif     (ilastik input)
    │   └── probs/                    vid{N}_maxmed_probs.h5 (ilastik output)
    ├── stitch/
    │   ├── positions.json            FULL-RES affine per tile
    │   └── provenance.json
    ├── fuse/                         feathered stitched images
    │   ├── intensity.tif             from std_clahe
    │   ├── maxmed.tif                from maxmed
    │   ├── probs_logit.tif           per-tile probs, logit-fused
    │   ├── probs_linear.tif          per-tile probs, arithmetic-fused
    │   └── provenance.json
    ├── mask/
    │   ├── mask.tif                  binary 0/255 vessel mask
    │   ├── mask.json                 threshold-tool sidecar (slider values)
    │   └── provenance.json
    └── graph/
        ├── graph.gpickle
        └── provenance.json

## Downsample model

Two independent factors, both recorded in `provenance.json`:

- **`stitch_downsample`** — display downsample used by the napari
  stitcher. `positions.json` is always at FULL RES (the stitcher
  multiplies back), so this factor does not leak downstream.
- **`fuse_downsample`** — resolution of the feathered stitched images
  and every downstream artifact (mask, graph). Graph node coords are
  in fused-pixel space; multiply by `fuse_downsample` to get full-res
  image coordinates.

Set both in `plexusflow.yaml`; defaults are 2 and 2.

## Bootstrap a new stage

    plexusflow init PlexusFlow21S

Creates the folder scaffold and a starter `plexusflow.yaml`. If the
tiles already live somewhere else (e.g. an existing PerTileFlow
`Somites21/Mosaic/` layout), edit the yaml:

    external_std_clahe_dir: ~/.../Somites21/Mosaic/StdFilter
    external_maxmed_dir:    ~/.../Somites21/Tiles/MaxMedImages
    external_probs_dir:     ~/.../Somites21/Tiles/Probs

Verify:

    plexusflow info PlexusFlow21S

## Run the pipeline

The same commands work for any embryo: everything stage-specific lives in the stage folder (`plexusflow.yaml`, the graph
and its tags).  Steps marked GUI are interactive; the rest are batch.

| # | step | command | writes |
|---|---|---|---|
| 1 | bootstrap | `plexusflow init STAGE` (then set `external_raw_video_dir`, `raw_fps` in `plexusflow.yaml`) | folder scaffold, yaml |
| 2 | tile projections | `plexusflow preprocess STAGE --which maxmed,bandpass` | `tiles/<projection>/` |
| 3 | vessel probabilities | ilastik on `tiles/maxmed/` (outside plexusflow) | `tiles/probs/` |
| 4 | stitch (GUI) | `plexusflow stitch STAGE` | `stitch/positions.json` |
| 5 | fuse | `plexusflow fuse STAGE` | `fuse/*.tif` |
| 6 | mask (GUI) | `plexusflow mask STAGE` (or `mve`: mask + vectorise + preview) | `mask/mask.tif` |
| 7 | vectorise | `plexusflow vectorize STAGE` | `graph/graph.gpickle` |
| 8 | edit graph (GUI) | `plexusflow edit STAGE` | `graph/graph.gpickle` |
| 9 | condense | `plexusflow condense STAGE` | `graph/graph_vessels.gpickle` |
| 10 | measure flow | `plexusflow analyze STAGE` (hours; `--tiles`, `--out` for a local copy) | `graph/graph_vessels_piv.gpickle`, `batch_kl_inputs/` |
| 11 | annotate (GUI) | `plexusflow workbench STAGE`: Edit tab → mark the dorsal-aorta anchors *source* and the vein anchors *sink*; View tab → region tools, *Tag class* → tag **open boundary** regions (beside the aortae) and **topology uncertain** regions (graph does not represent the vessels); Save | tags on `graph_vessels_piv.gpickle` |
| 12 | stage settings | add `ac_ref_tile` (a clearly beating tile near the aortae) and, if needed, `ac_f0_band`, `ac_veins`, `ac_gauge_block`, `f0_min_hz`/`f0_max_hz`, `of_dt` to `plexusflow.yaml` | |
| 13 | reconstruct | `bash scripts/run_stage.sh STAGE` (`--no-ac`, `--no-validate`) | `results/` |
| 14 | view | `plexusflow view STAGE` (read-only) | |

Step 13 runs `dc_reconcile.py` (reliability model fitted on this stage's junctions + mean-flow reconstruction), `dc_qc.py`
(review list `results/dc_qc.csv`: gross-error tests — measurement, repeat, nodal, radius — plus loop flow and large
unmeasured flows, grouped into hotspots ranked by the flow involved; for review in the workbench, never applied
automatically), the validations (`dc_validate_weighting.py`, `dc_junction_calibration.py`, `dc_withholding_curve.py`), `ac_field.py`
(tile-synchronised pulsatile field) and `mech/ac_estimate.py` (distensibility fit).  Set `PLEXUSFLOW_RESULTS` to write
results elsewhere.  The read-only viewer shows the measured graph and, when `results/` holds them, the reconstructed mean
flow and pulsatile field.  Region tags can be revised in the workbench at any time; rerun step 13 afterwards
(`scripts/dc_topology_sensitivity.py` compares the field with and without the topology regions).

**Editing an analysed graph.**  After changing the graph (step 8) on a stage that is already measured: `plexusflow condense
STAGE`, `plexusflow carry-measurements STAGE` (copies measurements of unchanged vessels), `plexusflow analyze STAGE
--skip-analyzed` (measures only edited or new vessels).

**Changing the flux estimator's section rule.**  `python3 scripts/kl_reaggregate.py` re-estimates every recording from
`batch_kl_inputs/` without re-running optical flow (backs up the graph first); then rerun step 13.

**Reproducing the paper.**  `bash scripts/reproduce_paper.sh` runs step 13 for the 21S release plus the benchmark and
figure scripts (README).

Each command has `--help`.  Common overrides:

    plexusflow stitch    STAGE --tile-source maxmed
    plexusflow fuse      STAGE --which probs --fusion linear
    plexusflow mask      STAGE --prob-source intensity --video foo.tif
    plexusflow vectorize STAGE --no-for-editor
    plexusflow edit      STAGE --from-mask

## Prerequisites

* The interactive stitcher and mask editor are PerTileFlow GUIs (not distributed here): set `PERTILEFLOW_ROOT` to a
  PerTileFlow checkout, or `PLEXUSFLOW_STITCH_SCRIPT` / `PLEXUSFLOW_THRESHOLD_SCRIPT` to the scripts.  Their outputs
  (`stitch/positions.json`, `mask/mask.tif`) are part of the data release, so the analysis does not need them.
* net3 (`pip install -e ".[vectorize]"`, from github.com/ajmayer95/net3) for vectorisation and the graph editor.
* napari, tifffile, scipy, numpy, scikit-image for the fuse/mask paths.

## Coordinate provenance

Because coordinates flow through multiple resolutions:

* `stitch/positions.json` — FULL-RES affines per tile
* `fuse/*.tif` — pixel grid at (full_res / fuse_downsample)
* `mask/mask.tif` — same grid as fuse
* `graph/graph.gpickle` — node x/y in that same grid

Downstream analyses (measurement, flow inference) can rescale by reading
`fuse_downsample` from `provenance.json`.
