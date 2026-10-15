"""Napari-based tile stitcher wrapper.

For now this shells out to the existing `scripts/plexus_stitch_v0.py` in
PerTileFlow. When the app is ported into plexusflow, this module will
grow the internals; the CLI surface here stays stable.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import StageConfig


def _locate_stitch_script() -> Path:
    """Find plexus_stitch_v0.py. Checks env var, then a couple of standard
    sibling paths. Users can override with $PLEXUSFLOW_STITCH_SCRIPT."""
    env = os.environ.get("PLEXUSFLOW_STITCH_SCRIPT")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.home() / "Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/"
                       "My Drive/Python/Yolk Sac Code/PerTileFlow/scripts/plexus_stitch_v0.py",
        Path(__file__).resolve().parents[4] / "PerTileFlow/scripts/plexus_stitch_v0.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "plexus_stitch_v0.py not found. Set $PLEXUSFLOW_STITCH_SCRIPT to its path.")


def launch_stitcher(cfg: StageConfig, tile_source: str = "std_clahe") -> int:
    """Open the napari stitcher on the stage's per-tile images.

    tile_source ∈ {'std_clahe', 'maxmed'} — which tile set to display.
    Positions are saved at FULL RES regardless of `stitch_downsample`.
    """
    cfg.ensure_dirs()
    script = _locate_stitch_script()
    if tile_source == "std_clahe":
        tile_dir = cfg.tiles_std_clahe_dir
        pattern = cfg.tile_std_clahe_pattern
    elif tile_source == "maxmed":
        tile_dir = cfg.tiles_maxmed_dir
        pattern = cfg.tile_maxmed_pattern
    else:
        raise ValueError(f'unknown tile_source: {tile_source}')

    if not tile_dir.exists():
        raise FileNotFoundError(f'tile dir does not exist: {tile_dir}')

    # Pass this stage's own tiles: the stitch script's built-in default
    # (--tiles 1-53) is the 21-somite tile count.
    tile_ids = cfg.discover_tile_ids(tile_source)
    if not tile_ids:
        raise FileNotFoundError(f'no tiles matching {pattern!r} in {tile_dir}')
    cmd = [sys.executable, str(script),
           "--image_dir", str(tile_dir),
           "--pattern", pattern,
           "--tiles", ",".join(str(t) for t in tile_ids),
           "--downsample", str(cfg.stitch_downsample),
           "--output", str(cfg.positions_path)]
    if cfg.positions_path.exists():
        cmd += ["--load-positions", str(cfg.positions_path)]
    print(f'[stitch] launching napari on {tile_dir}\n         '
          f'positions will save to {cfg.positions_path}')
    rc = subprocess.call(cmd)
    if rc == 0 and cfg.positions_path.exists():
        cfg.write_provenance("stitch", extra={
            "tile_source": tile_source,
            "tile_dir": str(tile_dir),
            "tile_pattern": pattern,
        })
    return rc
