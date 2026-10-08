# PlexusFlow21S — launch commands (copy/paste)

Stage root: `/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S`

Prefix every command with:

```bash
cd "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/Python/Yolk Sac Code/plexusflow/src"
```

## Read-only mosaic viewer

```bash
python3 -m plexusflow.cli view "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Threshold + paint mask editor (two sliders)

```bash
python3 -m plexusflow.cli mask "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Unified mask + vectorize + preview (NEW — one window)

```bash
python3 -m plexusflow.cli mve "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Graph editor (net3, on the current graph)

```bash
python3 -m plexusflow.cli edit "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Graph editor from mask (net3 edit-mask — vectorize + edit in one step)

```bash
python3 -m plexusflow.cli edit --from-mask "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Batch analyze (Farneback OF → per-tile Q on graph)

Full graph, all tiles:

```bash
python3 -m plexusflow.cli analyze "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S" --workers 4
```

Subset of tiles (comma or range):

```bash
python3 -m plexusflow.cli analyze "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S" --tiles 1,2,41,44,52,53 --workers 4
```

## Vectorize the current mask into a graph

```bash
python3 -m plexusflow.cli vectorize "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Condense degree-2 chains

```bash
python3 -m plexusflow.cli condense "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Downsample raw videos → videos_realtime bundle

```bash
python3 -m plexusflow.cli downsample-videos "/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/My Drive/PlexusFlow21S"
```

## Other commands (info)

- `plexusflow init <root>`  — scaffold a new stage folder
- `plexusflow info <root>`  — print discovered tile counts
- `plexusflow preprocess <root>`  — generate CLAHE / maxmed / probs
- `plexusflow stitch <root>`  — interactive stitcher
- `plexusflow fuse <root>`  — fuse tiles into a mosaic
- `plexusflow version`

See `python3 -m plexusflow.cli --help` for full list.
