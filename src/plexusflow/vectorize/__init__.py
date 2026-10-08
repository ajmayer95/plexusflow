"""Mask → graph via net3. Thin wrapper over `net3.cli`.

Coordinates in the resulting graph are in FUSED pixel coordinates
(the resolution of the mask). Multiply by `fuse_downsample` to get
full-res image coordinates.

Preprocessing: when ``cfg.vectorize_fill_holes > 0`` the mask is passed
through ``skimage.morphology.remove_small_holes`` before net3 sees it.
The fill is applied to a temp file so the user's ``mask.tif`` on disk
stays untouched — hole-fill preferences don't rewrite hand-painted
edits.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import StageConfig


def _prep_mask_for_net3(cfg: StageConfig) -> tuple[Path, dict]:
    """Return (path_to_feed_net3, provenance_dict).  Fills holes on a
    temp copy when ``cfg.vectorize_fill_holes > 0``; otherwise returns
    the raw ``mask.tif``.
    """
    if cfg.vectorize_fill_holes <= 0:
        return cfg.mask_path, {"mask_fill_holes": 0}
    import numpy as np
    import tifffile
    from skimage.morphology import remove_small_holes
    m = tifffile.imread(cfg.mask_path)
    b = (m > 0)
    # skimage ≥ 0.26 prefers ``max_size``; fall back to legacy kwarg on older.
    try:
        b_filled = remove_small_holes(b, max_size=int(cfg.vectorize_fill_holes))
    except TypeError:
        b_filled = remove_small_holes(b, area_threshold=int(cfg.vectorize_fill_holes))
    n_filled = int((b_filled & ~b).sum())
    tmp = Path(tempfile.mkstemp(prefix="mask_filled_", suffix=".tif")[1])
    tifffile.imwrite(tmp, (b_filled.astype(np.uint8) * 255), compression='zlib')
    print(f'[vectorize] filled {n_filled:,} px in holes ≤ {cfg.vectorize_fill_holes} '
          f'(mask.tif on disk unchanged)')
    return tmp, {"mask_fill_holes": int(cfg.vectorize_fill_holes),
                 "mask_fill_holes_pixels_added": n_filled}


def run_vectorize(cfg: StageConfig, for_editor: bool = True) -> int:
    cfg.ensure_dirs()
    if not cfg.mask_path.exists():
        raise FileNotFoundError(f'no mask at {cfg.mask_path}. Run `plexusflow mask` first.')
    mask_in, fill_prov = _prep_mask_for_net3(cfg)
    cmd = [sys.executable, "-m", "net3.cli", "vectorize",
           str(mask_in),
           "-o", str(cfg.graph_path),
           "--min-feature-size", str(cfg.vectorize_min_feature_size),
           "--prune-order", str(cfg.vectorize_prune_order)]
    if for_editor:
        cmd += ["--for-editor"]
    print(f'[vectorize] {mask_in.name} → {cfg.graph_path.name}')
    try:
        rc = subprocess.call(cmd)
    finally:
        if mask_in != cfg.mask_path:
            try: mask_in.unlink()
            except OSError: pass
    if rc == 0 and cfg.graph_path.exists():
        prov = {
            "min_feature_size": cfg.vectorize_min_feature_size,
            "prune_order": cfg.vectorize_prune_order,
            "for_editor": for_editor,
            "mask_downsample": cfg.fuse_downsample,
            "coords_to_fullres_factor": cfg.fuse_downsample,
        }
        prov.update(fill_prov)
        cfg.write_provenance("graph", extra=prov)
    return rc
