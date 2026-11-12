"""Tile positions from stitch/positions.json (the stitcher's output), as used by the figure scripts.

Copied from net3.edit.video_overlay.load_tile_positions so that figures do not need the optional net3 dependency.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict


def load_tile_positions(path: str | Path) -> Dict[int, dict]:
    """Parse a tile-positions JSON.  Returns ``{tile_id: {top_left_x,
    top_left_y, [scale_x, scale_y], [tile_width, tile_height]}}``.

    Accepts two common schemas:

    1. **Simple / flat**: top-level dict maps tile-id → position info::

           {"1": {"top_left_x": 0, "top_left_y": 0,
                    "tile_width": 640, "tile_height": 480}, ...}

    2. **PerTileFlow-style nested**: outer dict has a ``tiles`` key
       whose value is the per-tile mapping, with ``translate_x/y`` +
       ``scale_x/y`` (no explicit width/height — scaling is per-pixel)::

           {"reference_vid": 6,
            "tiles": {"6": {"translate_x": -593.0, "translate_y": -2494.2,
                             "scale_x": 1.0016, "scale_y": 1.0038}, ...},
            "relative_offsets": {...}}

    For the PerTileFlow schema, ``translate_x/y`` may be negative
    (arbitrary reference origin).  We normalize by subtracting the
    minimum so mosaic display coords start at (0, 0).
    """
    p = Path(path)
    with open(p) as f:
        raw = json.load(f)
    # Unwrap PerTileFlow nesting
    if (isinstance(raw, dict) and 'tiles' in raw
            and isinstance(raw['tiles'], dict)):
        raw = raw['tiles']
    out: Dict[int, dict] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (ValueError, TypeError):
            continue
        if not isinstance(v, dict):
            continue
        # Position (either schema; prefer top_left_ names, fall back to translate_)
        tx = v.get('top_left_x')
        if tx is None:
            tx = v.get('translate_x') or v.get('x') or v.get('origin_x') or 0
        ty = v.get('top_left_y')
        if ty is None:
            ty = v.get('translate_y') or v.get('y') or v.get('origin_y') or 0
        entry: dict = dict(top_left_x=float(tx), top_left_y=float(ty))
        # PerTileFlow: per-pixel scale factors
        if 'scale_x' in v:
            entry['scale_x'] = float(v['scale_x'])
        if 'scale_y' in v:
            entry['scale_y'] = float(v['scale_y'])
        # Simple: explicit tile size in mosaic px
        w = v.get('tile_width') or v.get('width') or v.get('w')
        h = v.get('tile_height') or v.get('height') or v.get('h')
        if w is not None:
            entry['tile_width'] = float(w)
        if h is not None:
            entry['tile_height'] = float(h)
        # Full 3x3 affine (plexusflow / rotated stitching). Off-diagonal
        # elements encode rotation — scale_x/y alone won't reproduce the
        # tile placement when tiles are rotated.
        A = v.get('affine_matrix')
        if A is not None:
            try:
                import numpy as _np
                entry['affine_matrix'] = _np.asarray(A, dtype=float)
            except Exception:
                pass
        out[tid] = entry
    # Global tile-offset normalization: shift so the leftmost *FOOTPRINT
    # CORNER* (not the translate) sits at (0, 0). With rotated tiles the
    # translate is the raw-frame origin, but the tile's actual bbox
    # extends into negative coords via the rotation.  Using the min
    # translate under-shifts and leaves the tile grid overhanging into
    # negative x/y relative to a fuse output that was cropped to the
    # positive-coord bbox.
    if out:
        import numpy as _np
        # Standard raw-frame corner set; scaled at video-load time
        RAW_H, RAW_W = 704.0, 640.0
        corners = _np.array([[0, 0], [0, RAW_W],
                              [RAW_H, RAW_W], [RAW_H, 0]], dtype=float)
        all_y = []; all_x = []
        for e in out.values():
            A = e.get('affine_matrix')
            if A is not None:
                w = corners @ A[:2, :2].T + A[:2, 2]
                all_y += list(w[:, 0]); all_x += list(w[:, 1])
            else:
                sy = float(e.get('scale_y', 1.0)); sx = float(e.get('scale_x', 1.0))
                ty = float(e['top_left_y']);       tx = float(e['top_left_x'])
                all_y += [ty, ty + RAW_H * sy]
                all_x += [tx, tx + RAW_W * sx]
        min_y = min(all_y); min_x = min(all_x)
        # Always shift: the fused canvas starts at the footprint's min corner
        # whether the stitch coordinates are negative (21S) or all positive
        # (27S: tiles at ~+10,000 px landed that far off when this ran only
        # for negative minima).
        if min_x != 0 or min_y != 0:
            for e in out.values():
                e['top_left_x'] -= min_x
                e['top_left_y'] -= min_y
                A = e.get('affine_matrix')
                if A is not None:
                    A2 = A.copy()
                    A2[0, 2] -= min_y
                    A2[1, 2] -= min_x
                    e['affine_matrix'] = A2
    return out
