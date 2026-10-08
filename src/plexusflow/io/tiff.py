"""TIFF stack loading + illumination-fade truncation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter1d


def load_tiff_stack(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a TIFF stack as a (T, H, W) array.

    ``path`` may be a single multi-page TIFF or a directory of one-frame
    TIFFs (sorted by name).  ``max_frames`` limits how many frames are
    loaded.  Output dtype defaults to float32.
    """
    path = Path(path)
    if path.is_dir():
        tiff_files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {path}")
        frames = []
        for i, f in enumerate(tiff_files):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(tifffile.imread(f))
        stack = np.stack(frames, axis=0)
    else:
        stack = tifffile.imread(path)
        if max_frames is not None:
            stack = stack[:max_frames]
    return stack.astype(dtype)


def cut_before_fade(
    stack: np.ndarray,
    *,
    p_drop: float = 0.75,
    p_back: float = 0.90,
    smooth: int = 15,
    backoff: int = 5,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """Truncate a video stack before its illumination fade-out.

    Detects the first frame whose mean intensity drops below
    ``p_drop × peak``, walks back to where mean intensity was still
    ``p_back × peak`` (start of the fade), and cuts ``backoff`` more
    frames earlier for safety.  Returns the (possibly shortened) stack
    and the cut index (or ``None`` if no fade was found).
    """
    T = stack.shape[0]
    m = stack.reshape(T, -1).mean(axis=1).astype(float)
    ms = uniform_filter1d(m, size=max(1, smooth), mode="nearest") if smooth > 1 else m
    vmax = float(ms.max()) + 1e-12
    thr_drop = vmax * p_drop
    thr_back = vmax * p_back
    idx_drop = np.where(ms <= thr_drop)[0]
    if idx_drop.size == 0:
        if verbose:
            print(f"  Fade detection: none (keeping all {T} frames)")
        return stack, None
    t_drop = int(idx_drop[0])
    i = t_drop
    while i > 0 and ms[i] < thr_back:
        i -= 1
    cut = max(0, i - backoff)
    if cut < 16:
        if verbose:
            print(f"  Fade detection: cut too early ({cut}), keeping all {T} frames")
        return stack, None
    if verbose:
        print(f"  Fade detection: {T} → {cut} frames (cut at {cut}, drop detected at {t_drop})")
    return stack[:cut], cut
