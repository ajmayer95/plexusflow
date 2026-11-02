"""Acquisition time of each tile recording.

The Photron camera writes a ``.cih`` header next to every recording with its record ``Date`` / ``Time`` (minute
resolution).  The camera's absolute date is a wrong default, but its clock runs monotonically through the scan, so
times *relative to the first recording* are exact to the minute.  Raw recordings are not part of a data release, so
``write_acquisition_times`` stores the parsed times in ``<stage>/acquisition_times.json`` and
``tile_acquisition_minutes`` prefers that file.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
from pathlib import Path

FILENAME = "acquisition_times.json"


def _recording_dir(cfg, tile):
    raw = os.path.expanduser(str(cfg.external_raw_video_dir or cfg.raw_video_dir or ""))
    c = sorted(glob.glob(os.path.join(raw, cfg.raw_video_dir_pattern.format(vid=tile))))
    return c[0] if c else None


def _parse_raw(cfg, tiles):
    """{tile: (header timestamp or None, first-frame file mtime or None)} from the raw recordings."""
    out = {}
    for t in tiles:
        d = _recording_dir(cfg, t)
        if d is None:
            continue
        cih = glob.glob(os.path.join(d, "*.cih")); hdr = None
        if cih:
            txt = open(cih[0], errors="ignore").read()
            md, mt = re.search(r"Date\s*:\s*(\d+)/(\d+)/(\d+)", txt), re.search(r"Time\s*:\s*(\d+):(\d+)", txt)
            if md and mt:
                hdr = _dt.datetime(*map(int, md.groups()), *map(int, mt.groups())).timestamp()
        fs = sorted(glob.glob(os.path.join(d, "*.tif")))[:1]
        out[int(t)] = (hdr, os.stat(fs[0]).st_mtime if fs else None)
    return out


def tile_acquisition_minutes(stage, cfg, tiles):
    """Minutes after the first recording for each tile in ``tiles`` ({} if unavailable).

    Uses the camera headers if every tile has one, else the first raw frame's file time for every tile (one clock for
    all tiles: the two clocks differ by a fixed offset; they agree to ≤ 1.1 min on 21S).  Reads
    ``<stage>/acquisition_times.json`` when present (data releases), the raw recordings otherwise."""
    tiles = [int(t) for t in tiles]
    p = Path(stage) / FILENAME
    if p.exists():
        J = json.loads(p.read_text())["tiles"]
        raw = {int(k): (v.get("header"), v.get("file")) for k, v in J.items()}
    else:
        raw = _parse_raw(cfg, tiles)
    hdr = {t: raw[t][0] for t in tiles if t in raw and raw[t][0] is not None}
    fil = {t: raw[t][1] for t in tiles if t in raw and raw[t][1] is not None}
    clock = hdr if len(hdr) == len(tiles) else fil
    if len(clock) != len(tiles):
        return {}
    t0 = min(clock.values())
    return {t: (clock[t] - t0) / 60.0 for t in tiles}


def write_acquisition_times(stage, cfg, tiles):
    """Parse the raw recordings once and store their times in ``<stage>/acquisition_times.json``."""
    raw = _parse_raw(cfg, tiles)
    J = dict(source="Photron .cih header Date/Time (minute resolution) and first raw frame file mtime; relative times only",
             tiles={str(t): dict(header=h, file=f) for t, (h, f) in sorted(raw.items())})
    p = Path(stage) / FILENAME
    p.write_text(json.dumps(J, indent=1))
    return p


# ── representative raw frame (Fig 3) ─────────────────────────────────────────────────────────────────────────────
REP_DIR = "raw_subset"


def representative_raw_frame(cfg, tile, step=10):
    """The raw frame of a tile's recording closest (mean squared difference) to the per-pixel median of every
    ``step``-th frame.  Returns (image float array, frame index 1-based, number of raw frames)."""
    import numpy as np
    import tifffile
    d = _recording_dir(cfg, tile)
    raw = sorted(glob.glob(os.path.join(d, "*.tif"))) if d else []
    if not raw:
        raise FileNotFoundError(f"no raw recording for tile {tile} under {cfg.raw_video_dir}")
    stack = np.stack([tifffile.imread(f).astype(float) for f in raw[::step]]); med = np.median(stack, axis=0)
    k = int(np.argmin([np.mean((fr - med) ** 2) for fr in stack]))
    img = stack[k]
    return (img.mean(-1) if img.ndim == 3 else img), step * k + 1, len(raw)


def load_representative_frame(stage, cfg, tile, step=10):
    """Representative raw frame of a tile: from the raw recording if available, else from the copy stored in a data
    release (``<stage>/raw_subset/loc<tile>_representative.tif`` + ``.json``)."""
    import numpy as np
    import tifffile
    p = Path(stage) / REP_DIR / f"loc{int(tile)}_representative.tif"
    d = _recording_dir(cfg, tile)
    if d is None and p.exists():
        meta = json.loads(p.with_suffix(".json").read_text())
        return tifffile.imread(str(p)).astype(float), int(meta["frame"]), int(meta["n_frames"])
    return representative_raw_frame(cfg, tile, step)


def write_representative_frame(stage, cfg, tile, step=10):
    import numpy as np
    import tifffile
    img, k, n = representative_raw_frame(cfg, tile, step)
    out = Path(stage) / REP_DIR; out.mkdir(parents=True, exist_ok=True)
    p = out / f"loc{int(tile)}_representative.tif"
    tifffile.imwrite(str(p), img.astype(np.float32))
    p.with_suffix(".json").write_text(json.dumps(dict(tile=int(tile), frame=k, n_frames=n, step=step,
                                                       rule="raw frame closest to the per-pixel median of every step-th frame")))
    return p
