"""Per-tile temporal projections computed from raw videos.

Each projection reduces a (T, H, W) time series → (H, W) 2D image, and
writes it to a tile-source directory inside the stage folder. Once these
exist, downstream stages (stitch, fuse) can consume them without
referring back to the raw videos.

Supported projections:

* ``maxmed``    — temporal max, then 3×3 spatial median filter (kills
                   single-pixel hot pixels). Anatomical / structural
                   channel. Semantically matches the ilastik training
                   input in most pipelines.
* ``bandpass``  — bandpass 1.5–4 Hz per-pixel time series, RMS. Clean
                   vessel-specific detector (rejects stripes and drift
                   via narrow bandwidth). Best for pulsatile vessels.
* ``std``       — plain temporal std. Baseline "any temporal variance"
                   detector.
* ``hp_std``    — highpass at 0.5 Hz then std. Same broad detector as
                   std but with drift and stripes attenuated.
* ``std_clahe`` — highpass std + CLAHE contrast enhancement + median
                   filter. Matches the legacy PerTileFlow StdFilter
                   output. Useful for high-contrast visualization.
"""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from typing import Callable
import numpy as np
import tifffile
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter, gaussian_filter

from ..config import StageConfig


# --------------------------------------------------------------------
# Individual projection functions — pure numpy/scipy, testable alone
# --------------------------------------------------------------------

def proj_max_median3(video: np.ndarray) -> np.ndarray:
    """Temporal max, then 3×3 spatial median. Anatomical channel."""
    return median_filter(video.max(axis=0), size=3).astype(np.float32)


def proj_bandpass_rms(video: np.ndarray, fs: float,
                      low_hz: float = 1.5, high_hz: float = 4.0,
                      order: int = 4) -> np.ndarray:
    """Bandpass each pixel then take RMS. Cardiac-flow detector."""
    T, H, W = video.shape
    sos = butter(order, [low_hz, high_hz], btype='band', fs=fs, output='sos')
    flat = video.reshape(T, -1).astype(np.float32)
    filtered = sosfiltfilt(sos, flat, axis=0)
    return np.sqrt(np.mean(filtered ** 2, axis=0)).reshape(H, W).astype(np.float32)


def proj_std(video: np.ndarray) -> np.ndarray:
    """Plain temporal std."""
    return video.std(axis=0).astype(np.float32)


def proj_highpass_std(video: np.ndarray, fs: float,
                       cutoff_hz: float = 0.5, order: int = 4) -> np.ndarray:
    """Highpass each pixel then std. Broad detector with drift rejected."""
    T, H, W = video.shape
    sos = butter(order, cutoff_hz, btype='high', fs=fs, output='sos')
    flat = video.reshape(T, -1).astype(np.float32)
    filt = sosfiltfilt(sos, flat, axis=0)
    return filt.std(axis=0).reshape(H, W).astype(np.float32)


def proj_std_clahe(video: np.ndarray, fs: float,
                    clahe_block_size: int = 127,
                    clahe_clip_limit: float = 3.0,
                    median_radius: int = 5,
                    percentile_clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Highpass std + CLAHE + median. Matches legacy StdFilter appearance."""
    try:
        from skimage.exposure import equalize_adapthist
        from skimage.util import img_as_ubyte
    except ImportError as e:
        raise RuntimeError(
            "proj_std_clahe requires scikit-image (skimage). "
            "Install with: pip install scikit-image") from e
    s = proj_highpass_std(video, fs=fs)
    # Percentile stretch
    lo, hi = np.percentile(s, percentile_clip)
    if hi <= lo: hi = lo + 1e-9
    s = np.clip((s - lo) / (hi - lo), 0, 1)
    # CLAHE
    s = equalize_adapthist(s, kernel_size=clahe_block_size,
                           clip_limit=clahe_clip_limit / 100.0)
    # Median smoothing
    if median_radius > 0:
        s = median_filter(s, size=2 * median_radius + 1)
    return (s * 65535).astype(np.uint16).astype(np.float32)


# Registry: projection name → callable + output dtype/units
PROJECTIONS: dict[str, tuple[Callable, str]] = {
    "maxmed":     (proj_max_median3,  "float32"),
    "bandpass":   (proj_bandpass_rms, "float32"),
    "std":        (proj_std,          "float32"),
    "hp_std":     (proj_highpass_std, "float32"),
    "std_clahe":  (proj_std_clahe,    "float32"),
}


# --------------------------------------------------------------------
# Batch runner: iterate over all tiles, save projections into stage dirs
# --------------------------------------------------------------------

def _stage_projection_dir(cfg: StageConfig, name: str) -> Path:
    """Where to write per-tile projections for a given projection name.

    Uses the existing external override if set (so it can point at legacy
    dirs), otherwise falls back to the stage folder under `tiles/<name>/`.
    """
    if name == "std_clahe" and cfg.external_std_clahe_dir is None:
        return cfg.root / "tiles" / "std_clahe"
    if name == "maxmed" and cfg.external_maxmed_dir is None:
        return cfg.root / "tiles" / "maxmed"
    # For everything else (bandpass, std, hp_std), always stage folder
    return cfg.root / "tiles" / name


def _projection_pattern(name: str, cfg: StageConfig) -> str:
    """Filename pattern for a given projection name — reuses the legacy
    pattern for maxmed/std_clahe, or `vid{vid}_<name>.tif` otherwise."""
    if name == "std_clahe":
        return cfg.tile_std_clahe_pattern
    if name == "maxmed":
        return cfg.tile_maxmed_pattern
    return f"vid{{vid}}_{name}.tif"


def preprocess_tiles(cfg: StageConfig,
                      which: Iterable[str] = ("maxmed", "bandpass"),
                      max_frames: int | None = None,
                      tile_ids: list[int] | None = None,
                      overwrite: bool = False,
                      verbose: bool = True) -> dict[str, Path]:
    """Compute per-tile projections from raw videos, write into stage dirs.

    Returns {projection_name: output_dir_path}.
    """
    which = list(which)
    for name in which:
        if name not in PROJECTIONS:
            raise ValueError(f'unknown projection {name!r}; '
                             f'valid: {sorted(PROJECTIONS)}')

    out_dirs = {name: _stage_projection_dir(cfg, name) for name in which}
    patterns = {name: _projection_pattern(name, cfg) for name in which}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if tile_ids is None:
        tile_ids = cfg.discover_raw_tile_ids()
    if not tile_ids:
        raise RuntimeError(f'no raw tiles found under {cfg.raw_video_dir}')

    for i, tid in enumerate(tile_ids):
        # Skip work when all outputs already exist and overwrite=False
        out_paths = {name: out_dirs[name] / patterns[name].format(vid=tid)
                      for name in which}
        if not overwrite and all(p.exists() for p in out_paths.values()):
            if verbose:
                print(f'  tile {tid}: all outputs exist, skip')
            continue

        try:
            frames = cfg.raw_tile_frames(tid)
        except FileNotFoundError as e:
            print(f'  tile {tid}: MISSING raw ({e})'); continue
        if max_frames is not None:
            frames = frames[:max_frames]
        if verbose:
            print(f'  tile {tid}: loading {len(frames)} frames '
                  f'({i+1}/{len(tile_ids)})')
        video = np.stack([tifffile.imread(f) for f in frames]).astype(np.float32)

        for name in which:
            out = out_paths[name]
            if out.exists() and not overwrite:
                continue
            fn, _dtype = PROJECTIONS[name]
            # Filter functions that take fs
            if name in ("bandpass", "hp_std", "std_clahe"):
                arr = fn(video, fs=cfg.raw_fps)
            else:
                arr = fn(video)
            # Save as uint16 with p1-p99 stretch for consistent dynamic range
            lo, hi = np.percentile(arr, [1, 99])
            if hi <= lo: hi = lo + 1e-9
            stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
            u16 = (stretched * 65535).astype(np.uint16)
            tifffile.imwrite(out, u16, compression='deflate')
        if verbose:
            print(f'  tile {tid}: wrote {list(out_paths.values())[0].parent.parent.name}/{{'
                  f'{",".join(which)}}}/vid{tid}_*.tif')

    return out_dirs
