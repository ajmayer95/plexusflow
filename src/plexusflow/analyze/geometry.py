"""Coordinate math for graph → tile-local mapping.

Frames involved (all in image conventions unless noted):

* **graph frame** — output of :mod:`plexusflow.vectorize`.  ``x`` is
  column (positive right); ``y`` is Cartesian y (positive **up**, origin
  at the bottom of the fused canvas).  Pixel size = ``fuse_downsample``
  raw pixels.
* **fuse canvas image frame** — matches the fused TIFF
  (``fuse/intensity.tif`` etc.).  ``(row, col)`` with row positive down.
  Related to the graph frame by ``row = H_fuse - y``, ``col = x``.
* **downsampled positions frame** — the fuse canvas placed inside a
  larger world with an ``(y0, x0)`` bbox offset (see
  :func:`plexusflow.fuse.feather.compute_world_bbox`).  This is the
  frame the fuser warps into.
* **full-res positions frame** — same as above scaled by
  ``fuse_downsample``.  This is what
  ``positions.json['tiles'][tid]['affine_matrix']`` operates in
  (translations at full-res).
* **tile-local frame** — pixels of the raw tile video ``(row, col)``.

The forward map ``tile → mosaic`` is A_full · [row_local, col_local, 1];
the inverse recovers tile-local coords from any point in the graph frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import numpy as np
import tifffile

from ..config import StageConfig
from ..fuse.feather import compute_world_bbox


@dataclass
class TileGeom:
    """Everything needed to project graph coords into one tile's video."""

    tile_id: int
    A_full: np.ndarray                 # 3x3 tile→full-res-positions
    A_full_inv: np.ndarray             # cached inverse
    video_shape: tuple[int, int]       # (H, W) of raw tile video


@dataclass
class MosaicGeom:
    """Whole-mosaic geometry needed to map a graph edge to any tile."""

    tiles: dict[int, TileGeom]
    fuse_shape: tuple[int, int]        # (H_ds, W_ds) of fused canvas
    y0: float                          # ds-frame bbox origin (row)
    x0: float                          # ds-frame bbox origin (col)
    downsample: int                    # fuse_downsample
    pad: int                           # bbox padding used by fuser


def tile_video_shape(cfg: StageConfig, tile_id: int) -> Optional[tuple[int, int]]:
    """Peek the (H, W) of one raw tile video by reading its first frame.

    Returns None if the tile folder or a frame is missing."""
    try:
        frames = cfg.raw_tile_frames(tile_id)
    except FileNotFoundError:
        return None
    if not frames:
        return None
    with tifffile.TiffFile(str(frames[0])) as tf:
        H, W = tf.pages[0].shape[:2]
    return (int(H), int(W))


def load_tile_geometry(cfg: StageConfig,
                        pad: int = 20,
                        video_shapes: Optional[dict[int, tuple[int, int]]] = None
                        ) -> MosaicGeom:
    """Build the full mosaic geometry for a stage.

    Reads ``positions.json`` (full-res affines), the fused canvas shape
    (to size the graph frame), and every tile video's shape (needed to
    compute the world bbox — same as the fuser does).  Returns a
    MosaicGeom carrying per-tile forward+inverse affines.
    """
    positions_path = cfg.positions_path
    if not positions_path.exists():
        raise FileNotFoundError(f'no positions.json at {positions_path}')
    blob = json.loads(positions_path.read_text())
    tile_info = blob['tiles']

    ds = int(cfg.fuse_downsample)

    # Load full-res affines
    A_full: dict[int, np.ndarray] = {}
    for tid_s, info in tile_info.items():
        A_full[int(tid_s)] = np.array(info['affine_matrix'], dtype=float)

    # Per-tile video shapes (needed for the bbox — must match what the
    # fuser saw when it warped this tile into the canvas).  If the caller
    # passed shapes explicitly (e.g. from an earlier peek), reuse.
    if video_shapes is None:
        video_shapes = {}
        for tid in A_full:
            shp = tile_video_shape(cfg, tid)
            if shp is not None:
                video_shapes[tid] = shp
    if not video_shapes:
        raise RuntimeError('no tile video shapes could be read; '
                           'check external_raw_video_dir in plexusflow.yaml')

    # Downsampled affines (fuser's frame)
    A_ds = {tid: A_full[tid].copy() for tid in A_full}
    for tid in A_ds:
        A_ds[tid][0, 2] /= ds
        A_ds[tid][1, 2] /= ds

    # World bbox in downsampled frame — matches feather.compute_world_bbox
    # exactly so graph coords land where the fuser drew pixels.  Feather
    # subsamples tiles at load time (`im[::ds, ::ds]`), so the shapes it
    # feeds compute_world_bbox are DOWNSAMPLED shapes.  We do the same
    # to reproduce the same bbox.
    ds_shapes = {tid: (H // ds, W // ds)
                 for tid, (H, W) in video_shapes.items()}
    y0, y1, x0, x1 = compute_world_bbox(
        ds_shapes,
        {tid: A_ds[tid] for tid in video_shapes},
        pad=pad)

    # Fuse canvas shape.  Prefer the actual fused TIFF if present so
    # any downstream rounding difference is caught.
    fuse_H = int(y1 - y0)
    fuse_W = int(x1 - x0)
    for fp in (cfg.fuse_maxmed_path, cfg.fuse_intensity_path):
        if fp.exists():
            with tifffile.TiffFile(str(fp)) as tf:
                fh, fw = tf.pages[0].shape[:2]
            if (fh, fw) != (fuse_H, fuse_W):
                # tolerate a 1-2 pixel round-off; error if off by more
                if abs(fh - fuse_H) > 2 or abs(fw - fuse_W) > 2:
                    raise RuntimeError(
                        f'fuse canvas mismatch: bbox={fuse_H}x{fuse_W} '
                        f'vs stored {fh}x{fw} in {fp.name}')
                fuse_H, fuse_W = fh, fw
            break

    tiles: dict[int, TileGeom] = {}
    for tid in video_shapes:
        A = A_full[tid]
        tiles[tid] = TileGeom(
            tile_id=int(tid),
            A_full=A,
            A_full_inv=np.linalg.inv(A),
            video_shape=video_shapes[tid])

    return MosaicGeom(
        tiles=tiles,
        fuse_shape=(fuse_H, fuse_W),
        y0=float(y0),
        x0=float(x0),
        downsample=ds,
        pad=int(pad))


def graph_to_tile(gx: float, gy: float, geom: MosaicGeom, tile_id: int
                   ) -> tuple[float, float]:
    """Map one graph-frame point (x, y) to a tile's local (row, col).

    Returns (row_local, col_local) — may be outside the video bounds if
    the point isn't within this tile's footprint.
    """
    tg = geom.tiles[tile_id]
    H_ds = geom.fuse_shape[0]
    ds = geom.downsample
    row_fuse = H_ds - gy
    col_fuse = gx
    row_mosaic = (row_fuse + geom.y0) * ds
    col_mosaic = (col_fuse + geom.x0) * ds
    row_l, col_l, _ = tg.A_full_inv @ np.array([row_mosaic, col_mosaic, 1.0])
    return float(row_l), float(col_l)


def graph_to_tile_batch(coords_xy: np.ndarray, geom: MosaicGeom,
                         tile_id: int) -> np.ndarray:
    """Vectorised version of :func:`graph_to_tile` for an (N, 2) array
    of ``[x, y]`` graph-frame coords.  Returns ``(N, 2)`` of
    ``[row_local, col_local]``."""
    coords_xy = np.asarray(coords_xy, dtype=float)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError('coords_xy must be (N, 2)')
    tg = geom.tiles[tile_id]
    H_ds = geom.fuse_shape[0]
    ds = geom.downsample
    row_fuse = H_ds - coords_xy[:, 1]
    col_fuse = coords_xy[:, 0]
    row_mosaic = (row_fuse + geom.y0) * ds
    col_mosaic = (col_fuse + geom.x0) * ds
    ones = np.ones_like(row_mosaic)
    homo = np.stack([row_mosaic, col_mosaic, ones], axis=1)
    local = homo @ tg.A_full_inv.T
    return local[:, :2]


def warp_mask_to_tile(mask_fuse: np.ndarray, geom: MosaicGeom,
                       tile_id: int) -> np.ndarray:
    """Warp a fuse-frame binary mask into the given tile's video frame.

    Uses the same affine chain as :func:`graph_to_tile` (bbox origin →
    ds-mosaic → full-res → tile), inverted, so that ``mask_out[r, c]``
    is True iff the fuse-mask pixel corresponding to tile pixel ``(r, c)``
    is True.  Nearest-neighbour sampling (order=0) to keep sharp edges.
    """
    from scipy.ndimage import affine_transform
    tg = geom.tiles[tile_id]
    H, W = tg.video_shape
    ds = geom.downsample
    # Output (tile) pixel (r_l, c_l) → input (fuse) pixel:
    #   [row_mos, col_mos, 1] = A_full @ [r_l, c_l, 1]
    #   [row_ds, col_ds] = [row_mos, col_mos] / ds
    #   [row_fuse, col_fuse] = [row_ds, col_ds] - [y0, x0]
    # So the combined linear map from tile → fuse is:
    A = tg.A_full
    M = A[:2, :2] / ds
    off = A[:2, 2] / ds - np.array([geom.y0, geom.x0])
    warped = affine_transform(mask_fuse.astype(np.float32), M, offset=off,
                                output_shape=(H, W), order=0,
                                cval=0.0, prefilter=False)
    return warped > 0.5


def graph_radius_to_tile_px(radius_graph: float, geom: MosaicGeom,
                             tile_id: int) -> float:
    """Convert an edge radius from graph pixels to tile-video pixels.

    Uses ``ds * |det(A)|**(1/2)`` = ds because the plexusflow affine is
    a pure rotation × uniform-scale (det = scale²).  For a strictly
    anisotropic affine this would need a directional formula; the
    plexusflow stitcher currently emits isotropic-only transforms.
    """
    tg = geom.tiles[tile_id]
    lin = tg.A_full[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(lin))))
    # ``scale`` maps tile-local pixels to full-res mosaic pixels
    # (typically 1.0 for napari orthogonal fits).  Full-res mosaic →
    # graph = 1 / downsample.  So local = graph * ds / scale.
    return radius_graph * geom.downsample / max(scale, 1e-9)
