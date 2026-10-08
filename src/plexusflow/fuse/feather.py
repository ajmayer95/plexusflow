"""Feather-blend per-tile images with saved affine positions.

The stitcher (`plexusflow stitch`) writes per-tile affine transforms at
FULL-RES world coordinates. This module warps each tile through its
affine into a single canvas at `full_res / downsample` and blends
overlaps with a distance-from-tile-edge weight kernel.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json
import numpy as np
import tifffile
from scipy.ndimage import affine_transform, distance_transform_edt


def compute_world_bbox(tile_shapes: dict[int, tuple[int, int]],
                        affines: dict[int, np.ndarray],
                        pad: int = 20) -> tuple[int, int, int, int]:
    """Compute the world bounding box that contains all warped tiles.

    Returns (y0, y1, x0, x1) in the downsampled affine's coordinate frame.
    """
    xs: list[float] = []
    ys: list[float] = []
    for tid, (H, W) in tile_shapes.items():
        A = affines[tid]
        corners = np.array([[0, 0, 1], [H, 0, 1], [0, W, 1], [H, W, 1]]).T
        world = A @ corners
        ys.extend(world[0]); xs.extend(world[1])
    y0 = int(np.floor(min(ys))) - pad
    y1 = int(np.ceil(max(ys))) + pad
    x0 = int(np.floor(min(xs))) - pad
    x1 = int(np.ceil(max(xs))) + pad
    return y0, y1, x0, x1


def feather_kernel_from_tile(shape: tuple[int, int], alpha: float = 0.7) -> np.ndarray:
    """Distance-from-edge kernel for one tile. Interior → 1, edge → 0."""
    H, W = shape
    edge_mask = np.ones((H, W), dtype=np.float32)
    edge_mask[[0, -1], :] = 0
    edge_mask[:, [0, -1]] = 0
    d = distance_transform_edt(edge_mask)
    d = d / max(float(d.max()), 1.0)
    return np.clip(d, 0, 1) ** float(alpha)


def _load_tile(path: Path, downsample: int) -> np.ndarray:
    im = tifffile.imread(path)
    if im.ndim == 3:
        # Take the middle slice for (T, Y, X); or first channel for (Y, X, C)
        if im.shape[-1] <= 4 and im.shape[-1] < im.shape[0]:
            im = im[..., 0]
        elif im.shape[0] < 20 and im.shape[0] < im.shape[-1]:
            im = im[im.shape[0] // 2]
        else:
            im = im.max(axis=0)
    return im[::downsample, ::downsample].astype(np.float32)


def _load_positions_and_affines(positions_path: Path, downsample: int
                                 ) -> tuple[list[int], dict[int, np.ndarray]]:
    """Return (tile_ids_in_order, {tid: 3x3 affine at downsampled coords})."""
    blob = json.loads(Path(positions_path).read_text())
    tiles_info = blob['tiles']
    affines: dict[int, np.ndarray] = {}
    tids: list[int] = []
    for tid_s, info in tiles_info.items():
        tid = int(tid_s)
        A_full = np.array(info['affine_matrix'], dtype=float)
        A_ds = A_full.copy()
        A_ds[0, 2] /= downsample   # translate stored at full-res
        A_ds[1, 2] /= downsample
        affines[tid] = A_ds
        tids.append(tid)
    tids.sort()
    return tids, affines


def _warp_into_canvas(im: np.ndarray, A: np.ndarray,
                      canvas_shape: tuple[int, int],
                      y0: int, x0: int) -> np.ndarray:
    """Warp `im` under affine `A` into a canvas of shape `canvas_shape`,
    origin (y0, x0). Bilinear interpolation."""
    A = A.copy()
    A[0, 2] -= y0
    A[1, 2] -= x0
    M_out_to_in = np.linalg.inv(A[:2, :2])
    off = -M_out_to_in @ A[:2, 2]
    return affine_transform(im, M_out_to_in, offset=off,
                             output_shape=canvas_shape,
                             order=1, cval=0.0, prefilter=False)


def _iter_tiles(positions_path: Path,
                tile_dir: Path, tile_pattern: str,
                downsample: int, loader: Callable[[Path, int], np.ndarray] = _load_tile,
                verbose: bool = True):
    """Yield (tid, im_downsampled, A_downsampled) for each tile with a file."""
    tids, affines = _load_positions_and_affines(positions_path, downsample)
    for tid in tids:
        fp = tile_dir / tile_pattern.format(vid=tid)
        if not fp.exists():
            if verbose:
                print(f'  tile {tid}: MISSING {fp.name}')
            continue
        yield tid, loader(fp, downsample), affines[tid]


def feather_intensity(positions_path: str | Path,
                       tile_dir: str | Path,
                       tile_pattern: str,
                       out_path: str | Path,
                       downsample: int = 2,
                       alpha: float = 0.7,
                       contrast_percentiles: tuple[float, float] = (1.0, 95.0),
                       dtype: str = "uint16",
                       per_tile_normalize: bool = True,
                       per_tile_percentiles: tuple[float, float] = (1.0, 95.0),
                       flatfield_sigma: float | None = 80.0,
                       verbose: bool = True) -> Path:
    """Feather intensity tiles (StdFilter or MaxMed) into a single stitched
    image.

    Cross-tile brightness uniformity is handled by two independent steps:

    * `per_tile_normalize` (default True): before warping, each tile is
      individually normalized to [0, 1] using its own
      `per_tile_percentiles` (default (1, 95)). This ensures each tile
      contributes a comparable dynamic range to the feather, so overlap
      seams don't show brightness discontinuities and a few very-bright
      pixels in one tile can't drag other tiles' contrast down.
      `flatfield_sigma` (default 80 px): if not None, subtract a
      Gaussian-blurred background of this sigma from each tile before
      the percentile normalization — kills illumination gradients within
      each tile.

    * `contrast_percentiles` (default (1, 95)): final display stretch
      applied to the stitched output. Same rationale — use a lower top
      percentile (95 rather than 99.95) so a few outlier bright pixels
      don't compress the vessel-brightness range of the rest of the map.

    Output range: contrast-stretched to `dtype`'s full range using the
    given percentiles. Uncovered pixels stay at 0.
    """
    from scipy.ndimage import gaussian_filter as _gf

    def _per_tile_prep(im: np.ndarray) -> np.ndarray:
        if not per_tile_normalize:
            return im
        x = im
        if flatfield_sigma is not None:
            x = x - _gf(x, sigma=float(flatfield_sigma))
        lo, hi = np.percentile(x, per_tile_percentiles)
        if hi <= lo: hi = lo + 1e-9
        return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)
    positions_path = Path(positions_path).expanduser()
    tile_dir = Path(tile_dir).expanduser()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: figure out canvas size
    tile_shapes: dict[int, tuple[int, int]] = {}
    affines_full: dict[int, np.ndarray] = {}
    _tids, affines_full = _load_positions_and_affines(positions_path, downsample)
    for tid, im, A in _iter_tiles(positions_path, tile_dir, tile_pattern,
                                    downsample, verbose=verbose):
        tile_shapes[tid] = im.shape
    if not tile_shapes:
        raise RuntimeError(f'No tiles found matching {tile_pattern} in {tile_dir}')
    y0, y1, x0, x1 = compute_world_bbox(tile_shapes,
                                          {t: affines_full[t] for t in tile_shapes})
    H_out, W_out = y1 - y0, x1 - x0
    if verbose:
        print(f'canvas: {H_out} × {W_out} px (downsample={downsample})')

    num = np.zeros((H_out, W_out), dtype=np.float64)
    den = np.zeros((H_out, W_out), dtype=np.float64)

    for tid, im, A in _iter_tiles(positions_path, tile_dir, tile_pattern,
                                    downsample, verbose=False):
        im = _per_tile_prep(im)
        warped = _warp_into_canvas(im, A, (H_out, W_out), y0, x0)
        kern = feather_kernel_from_tile(im.shape, alpha=alpha)
        warped_w = _warp_into_canvas(kern, A, (H_out, W_out), y0, x0)
        num += warped * warped_w
        den += warped_w
        if verbose:
            print(f'  tile {tid}: warped', end='\r')

    out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    covered = out > 0
    if covered.any():
        lo, hi = np.percentile(out[covered], contrast_percentiles)
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-9
    stretched = np.clip((out - lo) / (hi - lo), 0, 1)
    if dtype == "uint16":
        out_arr = (stretched * 65535).astype(np.uint16)
    elif dtype == "uint8":
        out_arr = (stretched * 255).astype(np.uint8)
    else:
        out_arr = stretched.astype(np.float32)
    tifffile.imwrite(out_path, out_arr, compression='deflate')
    if verbose:
        print(f'\n→ {out_path.name}  ({out_arr.shape}, {out_arr.dtype})')
    return out_path


def feather_probabilities(positions_path: str | Path,
                           probs_dir: str | Path,
                           probs_pattern: str,
                           probs_channel: int,
                           out_path: str | Path,
                           fusion: str = "logit",     # or "linear"
                           downsample: int = 2,
                           alpha: float = 0.7,
                           verbose: bool = True) -> Path:
    """Feather per-tile ilastik probability outputs.

    fusion = 'linear': weighted arithmetic mean of probabilities.
    fusion = 'logit':  weighted mean in logit space, then sigmoid back.
                       Correct if tile classifiers are independent.
    """
    import h5py

    def _load_prob(path: Path, ds: int) -> np.ndarray:
        with h5py.File(path, 'r') as f:
            p = f['exported_data'][:, :, probs_channel].astype(np.float32)
        p = p[::ds, ::ds]
        if fusion == "logit":
            eps = 1e-4
            pc = np.clip(p, eps, 1 - eps)
            return np.log(pc / (1.0 - pc)).astype(np.float32)
        return p

    positions_path = Path(positions_path).expanduser()
    probs_dir = Path(probs_dir).expanduser()
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_shapes: dict[int, tuple[int, int]] = {}
    _tids, affines_full = _load_positions_and_affines(positions_path, downsample)
    for tid, im, A in _iter_tiles(positions_path, probs_dir, probs_pattern,
                                    downsample, loader=_load_prob, verbose=verbose):
        tile_shapes[tid] = im.shape
    if not tile_shapes:
        raise RuntimeError(f'No prob tiles found matching {probs_pattern} in {probs_dir}')
    y0, y1, x0, x1 = compute_world_bbox(tile_shapes,
                                          {t: affines_full[t] for t in tile_shapes})
    H_out, W_out = y1 - y0, x1 - x0
    if verbose:
        print(f'canvas: {H_out} × {W_out} (downsample={downsample}, fusion={fusion})')

    num = np.zeros((H_out, W_out), dtype=np.float64)
    den = np.zeros((H_out, W_out), dtype=np.float64)

    for tid, im, A in _iter_tiles(positions_path, probs_dir, probs_pattern,
                                    downsample, loader=_load_prob, verbose=False):
        warped = _warp_into_canvas(im, A, (H_out, W_out), y0, x0)
        kern = feather_kernel_from_tile(im.shape, alpha=alpha)
        warped_w = _warp_into_canvas(kern, A, (H_out, W_out), y0, x0)
        num += warped * warped_w
        den += warped_w
        if verbose:
            print(f'  tile {tid}: warped', end='\r')

    mean_val = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    if fusion == "logit":
        out = 1.0 / (1.0 + np.exp(-mean_val))
        out = np.where(den > 1e-9, out, 0.0)   # uncovered → 0, not sigmoid(0)=0.5
    else:
        out = mean_val
    out_u16 = (np.clip(out, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(out_path, out_u16, compression='deflate')
    if verbose:
        covered = out[den > 1e-9]
        if covered.size:
            print(f'\n→ {out_path.name}  ({out_u16.shape}) '
                  f'covered stats: median={float(np.median(covered)):.3f}, '
                  f'frac>0.9={100*(covered > 0.9).mean():.1f}%')
    return out_path
