"""Shared whole-organ mosaic for the paper figures (Fig 1c and Fig 2a use exactly this image and field of view).

fuse/organ_fusion.tif; only the exterior (outside the filled tissue outline) is masked.  Display only: saturated (255) blobs
and zero-valued holes inside the tissue are filled by a masked Gaussian average of the surrounding tissue (normalised
convolution, σ = 6 px) — state in the captions.  The embryo is part of the image; it is excluded from the vessel analysis
downstream (curated mask), not here.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter


def load_mosaic(stage: Path):
    """→ dict(disp, outside, vmin, vmax, extent) with extent = (x0, x1, y1, y0) the shared field of view (pixel bounds)."""
    mos = tifffile.imread(str(Path(stage) / "fuse" / "organ_fusion.tif")).astype(float)
    mos = mos.mean(-1) if mos.ndim == 3 else mos
    outside = ~ndi.binary_fill_holes(mos > 0)
    bad = ~outside & ((mos >= mos.max()) | (mos <= 0)); bad = ndi.binary_dilation(bad, iterations=1) & ~outside
    good = (~bad & ~outside).astype(float); num = gaussian_filter(mos * good, 6.0); den = gaussian_filter(good, 6.0)
    disp = np.where(bad, num / np.maximum(den, 1e-6), mos)
    ok = ~outside & ~bad
    return dict(disp=disp, outside=outside, vmin=float(np.percentile(disp[ok], 1)), vmax=float(np.percentile(disp[ok], 99.5)),
                fill_fraction=float(bad.sum() / (~outside).sum()), extent=(0, mos.shape[1], mos.shape[0], 0))


def show_mosaic(ax, M):
    ax.imshow(np.ma.masked_where(M["outside"], M["disp"]), cmap="gray", vmin=M["vmin"], vmax=M["vmax"], interpolation="antialiased")
    x0, x1, y1, y0 = M["extent"]; ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
