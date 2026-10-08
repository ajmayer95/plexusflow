# plexusflow pipeline (folder → graph)

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

    plexusflow stitch    PlexusFlow21S              # napari GUI
    plexusflow fuse      PlexusFlow21S              # batch — all three
    plexusflow mask      PlexusFlow21S              # napari GUI
    plexusflow vectorize PlexusFlow21S              # batch
    plexusflow edit      PlexusFlow21S              # napari GUI

Each command has --help and accepts overrides. Common ones:

    plexusflow stitch    PlexusFlow21S --tile-source maxmed
    plexusflow fuse      PlexusFlow21S --which probs --fusion linear
    plexusflow mask      PlexusFlow21S --prob-source intensity --video foo.tif
    plexusflow vectorize PlexusFlow21S --no-for-editor
    plexusflow edit      PlexusFlow21S --from-mask

## Prerequisites

* PerTileFlow's `scripts/plexus_stitch_v0.py` and `plexus_threshold.py`
  reachable (auto-detected in the default location; override via
  `PLEXUSFLOW_STITCH_SCRIPT` and `PLEXUSFLOW_THRESHOLD_SCRIPT`).
* net3 installed with the [gui] extra for the graph editor.
* napari, tifffile, scipy, numpy, scikit-image for the fuse/mask paths.

## Coordinate provenance

Because coordinates flow through multiple resolutions:

* `stitch/positions.json` — FULL-RES affines per tile
* `fuse/*.tif` — pixel grid at (full_res / fuse_downsample)
* `mask/mask.tif` — same grid as fuse
* `graph/graph.gpickle` — node x/y in that same grid

Downstream analyses (measurement, flow inference) can rescale by reading
`fuse_downsample` from `provenance.json`.
