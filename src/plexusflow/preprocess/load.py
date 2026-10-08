"""Load raw tile folders (Photron FASTCAM output).

A raw tile is a folder of individual TIFF frames named
``<basename>NNNNNN.tif``, plus sidecar ``.cih`` (camera header) etc.  Frame
count per tile is typically ~1600 at 250 fps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _sorted_frame_files(tile_folder: Path) -> list[Path]:
    """Return the tile's frame TIFFs in acquisition order."""
    folder = Path(tile_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"tile folder not found: {folder}")
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"no .tif frames in {folder}")
    return files


def load_frame_range(
    tile_folder: Path,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> np.ndarray:
    """Load a slice of frames as a ``(T, H, W)`` float32 array.

    Parameters
    ----------
    tile_folder
        Path to the tile's raw frame folder.
    start, stop, step
        Standard slice-style indexing into the sorted frame list.  ``stop``
        of ``None`` reads to the end.
    """
    files = _sorted_frame_files(Path(tile_folder))
    files = files[start:stop:step]
    if not files:
        raise ValueError(f"empty slice {start}:{stop}:{step} on {tile_folder}")
    return np.stack([tifffile.imread(str(f)) for f in files]).astype(np.float32)


def load_raw_tile(
    tile_folder: Path,
    n_frames: int | None = None,
) -> np.ndarray:
    """Load a tile as ``(T, H, W)`` float32.

    Parameters
    ----------
    tile_folder
        Path to the tile's raw frame folder.
    n_frames
        If given, subsample uniformly to at most this many frames
        (preserves temporal spread).  ``None`` loads all frames.
    """
    files = _sorted_frame_files(Path(tile_folder))
    if n_frames is not None and n_frames < len(files):
        idxs = np.linspace(0, len(files) - 1, n_frames).astype(int)
        files = [files[i] for i in idxs]
    return np.stack([tifffile.imread(str(f)) for f in files]).astype(np.float32)
