"""Batch Farneback-OF over a plexusflow vessel graph.

Streamlined port of PerTileFlow's ``scripts/batch_analyze_vessels.py``:

* Consumes the *vessel graph* produced by
  :func:`plexusflow.analyze.condense.condense_stage` — one edge per
  vessel, ``path`` polyline in mosaic graph coords.
* For each tile with a raw video, runs Farneback dense OF once, then
  for every vessel whose polyline lands (partially) inside that tile's
  field of view, integrates the parallel component into a per-frame
  flow signal Q(t) and fits harmonics.
* Writes ``measurements_piv[tile_id]`` on each vessel edge and saves
  the graph incrementally after every tile (so a crash mid-run keeps
  progress).

The coordinate transform lives in :mod:`plexusflow.analyze.geometry`
and handles the full affine (including napari's rotation) plus
downsample+bbox that separate the graph frame from tile pixels.

All the OF-profile + harmonic-fit + Kloosterman code lives inside
plexusflow itself (``plexusflow.analyze.{profile, centerline,
kloosterman, harmonic}`` + ``plexusflow.io.tiff``).  No pertile
dependency.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import networkx as nx
import tifffile

from ..config import StageConfig
from .geometry import (MosaicGeom, load_tile_geometry,
                         graph_to_tile_batch, graph_radius_to_tile_px,
                         tile_video_shape, warp_mask_to_tile)

MIN_ARC_PX = 8   # minimum in-FOV arc length (tile pixels) worth analyzing

# Trim this much arc-length off each end of the centerline before sampling
# R_edt / computing the OF profile.  Junction pixels sit in wider mask
# regions (multiple vessels meeting), so EDT samples there are inflated
# and give a fake widening at the endpoints.  Same trick as legacy
# PerTileFlow's MARGIN_PX.
END_TRIM_TILE_PX = 5.0

# Farneback params — same as legacy batch_analyze_vessels.  Kept close
# to defaults because these are the values validated against real data.
# New: use pertile's mask-gated parallel-ridge refinement so the CL
# doesn't wander into the wrong vessel or an avascular noise ridge.
FB_KWARGS = dict(
    # dt=1: reverted from dt=3 (2026-08-24) after direct verification that
    # dt=3 ALIASES fast arterial vessels through Farneback's winsize=3
    # capture range (~1-2 px).  At v~600-1000 µm/s on A vessels: dt=1
    # gives 1.5 px/frame displacement (in range); dt=3 gives 4-6 px/frame
    # (aliased flat-topped → mean v under-estimated by ~2.5×).  A=22658
    # tile 14 measured 617 µm/s at dt=1 vs 245 µm/s at dt=3.  dt=1 is
    # noisier for slow venous vessels but UNBIASED — bias is worse than
    # noise for the boundary vessels the sim depends on.
    bg_percentile=10, fb_winsize=3, fb_levels=1, fb_iterations=3,
    fb_poly_n=3, fb_poly_sigma=0.7, dt=1, n_bins=30, fit_fraction=0.75,
    refine_centerline=True, use_flow_tangent=True,
    tangent_smooth_sigma=5.0,
    # ridge_iterations=1 → the ridge tracker runs max(2,1+1)=2 passes, IDENTICAL
    # to the viewer's fresh-OF (which passes no ridge_iterations → compute default
    # 1).  This alignment is REQUIRED: the ridge tracker does not converge (more
    # passes drift the CL back toward the graph CL and flip Kloosterman validity),
    # so batch=8/fresh=1 gave up to 150% Q differences.  refinement_mode and
    # ridge_ang_tol_deg are dead params (# noqa ARG001 in compute) — kept only for
    # signature compatibility; they have no effect.
    refinement_mode="parabola_peak",     # IGNORED (ARG001) — kept for compat
    ridge_ang_tol_deg=35.0,               # IGNORED (ARG001)
    ridge_iterations=1,                   # → 2 passes, matches fresh-OF exactly
    ridge_shift_tol_px=1.0)

# QC thresholds — a measurement fails if ANY of these fires.  Kept
# conservative; downstream can re-filter with tighter thresholds.
QC_MIN_PEAK_UMPS = 15.0        # coherent flow floor (µm/s)
QC_MIN_PEAK_TO_RMS = 1.00      # peak-vs-noise; theoretical Poiseuille ideal is
                                # √(15/8)≈1.37 on ρ∈[-R,R] but real vessels sit
                                # 1.24-2.45 (p10-p90).  1.05 never fired on the
                                # full-53-tile sweep → dropped to 1.00 (effectively
                                # off; keep as safety net against pure-noise runs).
QC_MAX_PEAK_OFFSET_FRAC = 1.0  # |ρ_peak| < 1.0·R_edt_mean (peak within a radius is fine
                                # — real curved vessels naturally shift)
QC_MAX_ASYMMETRY = 8.0         # left/right energy ratio (loosened; catches septated vessels)
QC_MAX_REFINED_KINK_DEG = 45.0 # refined-CL mean successive-tangent angle change.
                                # Was 35° → dropped 29% of measurements on the
                                # dense hole-filled graph (75th pct = 38° for the
                                # full 53-tile sweep).  45° saves ~14% while still
                                # catching genuine refinement-noise cases.
QC_MAX_KINK_RATIO = 1.5        # refined/initial kink; refinement made CL worse
QC_MIN_ASPECT = 0.4            # body_arc / (2·R); dense-plexus branches are often
                                # short.  Was 0.5 → dropped ~5% of otherwise-fine
                                # short segments.  0.4 keeps 2R·0.4 ≈ 0.8R arc
                                # (still enough for OF).


def _kink_deg(cl):
    """Mean successive tangent-angle change along a centerline, in deg.
    Legit vessels sit at 5–15°; refinement noise blows this past 30°."""
    d = np.diff(np.asarray(cl, float), axis=0)
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 0], d[:, 1])
    return float(np.mean(np.abs(np.diff(np.unwrap(ang))))) * 180.0 / np.pi


def _profile_qc(bin_centers, profile_v, R_edt_mean_um, px_um, dt_s,
                init_kink_deg=float("nan"), refined_kink_deg=float("nan"),
                aspect_ratio=float("nan")):
    """Compute QC on an axial OF profile.  Returns dict with metrics +
    passed_qc bool + reason string.

    Extra CL-geometry gates (kink + aspect ratio) catch pathologies the
    profile-shape gates miss — paired A/V bundles, junction pockets, and
    vessels where refinement fails because the profile has no coherent
    peak per slice."""
    nan = float("nan")
    empty_extras = dict(cl_kink_init_deg=init_kink_deg,
                         cl_kink_refined_deg=refined_kink_deg,
                         aspect_ratio=aspect_ratio)
    if len(bin_centers) == 0 or np.all(~np.isfinite(profile_v)):
        return dict(passed_qc=False, reason="empty_profile",
                     peak_v_ums=nan, peak_offset_um=nan,
                     peak_to_rms=nan, asymmetry_ratio=nan, **empty_extras)
    rho_um = np.asarray(bin_centers) * px_um
    v_ums = np.asarray(profile_v) * px_um / dt_s
    v_abs = np.abs(v_ums)
    if not np.any(np.isfinite(v_abs)):
        return dict(passed_qc=False, reason="all_nan",
                     peak_v_ums=nan, peak_offset_um=nan,
                     peak_to_rms=nan, asymmetry_ratio=nan, **empty_extras)
    peak_v = float(np.nanmax(v_abs))
    peak_j = int(np.nanargmax(v_abs))
    peak_offset_um = float(abs(rho_um[peak_j]))
    v_rms = float(np.sqrt(np.nanmean(v_ums ** 2)))
    peak_to_rms = peak_v / max(v_rms, 1e-9)
    left = v_abs[rho_um < 0]; right = v_abs[rho_um > 0]
    lE = float(np.nansum(left ** 2)); rE = float(np.nansum(right ** 2))
    asymm = max(lE, rE) / max(min(lE, rE), 1e-9)
    reasons = []
    if peak_v < QC_MIN_PEAK_UMPS:
        reasons.append(f"peak<{QC_MIN_PEAK_UMPS:g}umps")
    if peak_to_rms < QC_MIN_PEAK_TO_RMS:
        reasons.append(f"peak/rms<{QC_MIN_PEAK_TO_RMS:g}")
    R_ref = max(R_edt_mean_um if np.isfinite(R_edt_mean_um) else 0, 5.0)
    if peak_offset_um > QC_MAX_PEAK_OFFSET_FRAC * R_ref:
        reasons.append(f"peak_offset>{QC_MAX_PEAK_OFFSET_FRAC:g}R")
    if asymm > QC_MAX_ASYMMETRY:
        reasons.append(f"asym>{QC_MAX_ASYMMETRY:g}")
    if np.isfinite(refined_kink_deg) \
            and refined_kink_deg > QC_MAX_REFINED_KINK_DEG:
        reasons.append(f"kink>{QC_MAX_REFINED_KINK_DEG:g}deg")
    if np.isfinite(refined_kink_deg) and np.isfinite(init_kink_deg) \
            and init_kink_deg > 1e-3 \
            and refined_kink_deg / init_kink_deg > QC_MAX_KINK_RATIO:
        reasons.append(
            f"kink_ratio>{QC_MAX_KINK_RATIO:g}")
    if np.isfinite(aspect_ratio) and aspect_ratio < QC_MIN_ASPECT:
        reasons.append(f"aspect<{QC_MIN_ASPECT:g}")
    return dict(passed_qc=(len(reasons) == 0),
                 reason=(";".join(reasons) if reasons else "ok"),
                 peak_v_ums=peak_v, peak_offset_um=peak_offset_um,
                 peak_to_rms=peak_to_rms, asymmetry_ratio=asymm,
                 cl_kink_init_deg=init_kink_deg,
                 cl_kink_refined_deg=refined_kink_deg,
                 aspect_ratio=aspect_ratio)


# ---------------------------------------------------------------------
# Deferred bundle of analysis-pipeline callables + constants.
# Historically these came from PerTileFlow; they now live in plexusflow's
# own analyze/ + io/ modules.  Kept behind a single-call bundle helper
# so callers pass one ``pt`` dict rather than re-importing everywhere.
# ---------------------------------------------------------------------

def _import_pertile():
    """Return the analysis pipeline bundle.

    Name kept for backwards compatibility with older call sites (was
    the PerTileFlow imports collector).  All references now resolve
    within plexusflow — no pertile.* import happens.
    """
    from .profile import compute_of_profile_and_Q, _rebin_profile
    from .harmonic import fit_harmonics
    from .constants import PX_SIZE_UM, FRAME_DT_S, N_HARMONICS
    from ..io.tiff import load_tiff_stack, cut_before_fade
    return dict(
        compute_of_profile_and_Q=compute_of_profile_and_Q,
        _rebin_profile=_rebin_profile,
        fit_harmonics=fit_harmonics,
        PX_SIZE_UM=PX_SIZE_UM, FRAME_DT_S=FRAME_DT_S,
        N_HARMONICS=N_HARMONICS,
        load_tiff_stack=load_tiff_stack,
        cut_before_fade=cut_before_fade,
    )


# ---------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------

def _trim_endpoints(cl: np.ndarray, trim_px: float) -> np.ndarray:
    """Remove arc-length `trim_px` from each end of the centerline.

    Uses cumulative arc-length so the trim is a real distance, not a
    fraction of the point count.  Returns the interior slice.  If the
    resulting arc would be < 3 samples, returns the original array
    (better to keep something than nothing)."""
    if len(cl) < 4 or trim_px <= 0:
        return cl
    d = np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1]))
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]
    if total <= 2 * trim_px:
        return cl              # too short to trim safely
    lo = np.searchsorted(s, trim_px)
    hi = np.searchsorted(s, total - trim_px)
    if hi - lo < 3:
        return cl
    return cl[lo:hi]


def _find_body_region(cl: np.ndarray, edt: np.ndarray,
                       min_trim_px: float = 2.0,
                       tol_frac: float = 0.20,
                       max_trim_frac: float = 0.35) -> np.ndarray:
    """Adaptive endpoint trim: walk inward from each end until R_edt
    matches the vessel body (median of the middle 50% of R_edt) within
    `tol_frac`.  Always trims at least ``min_trim_px`` arc-length as a
    baseline so pixel-scale junction bulges are always removed.

    Uses the vessel's own R_edt profile to detect "junction bulges" —
    R_edt spikes at endpoints because those pixels sit at the confluence
    of multiple vessels.  Retains more of the vessel body than a fixed
    trim, and trims more when the bulge is broad.
    """
    if len(cl) < 6 or edt is None:
        return _trim_endpoints(cl, min_trim_px)
    H, W = edt.shape
    # Sample R_edt along cl (nearest neighbour is fine — cl is dense)
    rows = np.clip(np.round(cl[:, 0]).astype(int), 0, H - 1)
    cols = np.clip(np.round(cl[:, 1]).astype(int), 0, W - 1)
    R_edt = edt[rows, cols]
    if not np.any(R_edt > 0):
        return _trim_endpoints(cl, min_trim_px)
    q1, q3 = np.percentile(R_edt, [25, 75])
    body_mask = (R_edt >= q1) & (R_edt <= q3)
    body_R = float(np.median(R_edt[body_mask])) if body_mask.any() else float(np.median(R_edt))
    tol = tol_frac * body_R
    # Walk inward from each end while R_edt exceeds body + tol.
    lo = 0
    while lo < len(R_edt) - 4 and R_edt[lo] > body_R + tol:
        lo += 1
    hi = len(R_edt) - 1
    while hi > lo + 4 and R_edt[hi] > body_R + tol:
        hi -= 1
    # Cap the adaptive trim so we don't lose more than max_trim_frac of
    # the original arc — protects short vessels where a large fraction
    # of the CL would otherwise be treated as "junction bulge".
    if max_trim_frac is not None and 0 < max_trim_frac < 1:
        n_orig = len(cl)
        n_cap = max(3, int(round(n_orig * (1 - max_trim_frac))))
        if hi + 1 - lo < n_cap:
            # symmetric expand to reach n_cap
            need = n_cap - (hi + 1 - lo)
            left_expand = need // 2
            right_expand = need - left_expand
            lo = max(0, lo - left_expand)
            hi = min(n_orig - 1, hi + right_expand)
    cl_adaptive = cl[lo:hi + 1]
    return _trim_endpoints(cl_adaptive, min_trim_px)


def _largest_inbounds_segment(tile_rc: np.ndarray, H: int, W: int,
                                margin: int = 2) -> np.ndarray:
    """Return the longest contiguous subarray of `tile_rc` that stays
    inside [margin, H-margin] × [margin, W-margin].  Tile OF is only
    valid where the video actually covers, and we don't want to spend
    OF cycles on the parts of a vessel that sit outside this tile."""
    ib = ((tile_rc[:, 0] >= margin) & (tile_rc[:, 0] <= H - margin)
          & (tile_rc[:, 1] >= margin) & (tile_rc[:, 1] <= W - margin))
    if np.all(ib):
        return tile_rc
    if not np.any(ib):
        return tile_rc[:0]
    ch = np.diff(ib.astype(int))
    starts = list(np.where(ch == 1)[0] + 1)
    ends = list(np.where(ch == -1)[0] + 1)
    if ib[0]: starts = [0] + starts
    if ib[-1]: ends = ends + [len(ib)]
    if not starts or not ends:
        return tile_rc[:0]
    k = min(len(starts), len(ends))
    best = int(np.argmax(np.array(ends[:k]) - np.array(starts[:k])))
    return tile_rc[starts[best]:ends[best]]


def _in_fov_arc(path_xy: np.ndarray, geom: MosaicGeom, tile_id: int,
                 video_H: int, video_W: int) -> float:
    """Return in-FOV arc length (in tile pixels).  Cheap pre-filter."""
    tc = graph_to_tile_batch(path_xy, geom, tile_id)
    # graph_to_tile_batch returns (row, col); swap to (row, col) format
    seg = _largest_inbounds_segment(tc, video_H, video_W)
    if len(seg) < 5:
        return 0.0
    return float(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])).sum())


def _path_a_to_b(G: nx.Graph, A, B, path, step_px: float = 1.0) -> np.ndarray:
    """Edge path (graph coords) ordered A→B and resampled to ~``step_px``.

    `_analyze_vessel` signs flow along PATH order but labels the result
    A→B, so a path stored B→A (e.g. net3 `move_node` before 2026-10-07)
    silently flips the measured sign.  Resampling first means short
    straight paths (2 vertices after a node edit) survive the ≥5-point
    in-FOV segment check — the same resample the viewer's fresh OF does.
    """
    p = np.asarray(path, dtype=float)[:, :2]
    xa = np.array([G.nodes[A]["x"], G.nodes[A]["y"]], dtype=float)
    xb = np.array([G.nodes[B]["x"], G.nodes[B]["y"]], dtype=float)
    if np.linalg.norm(p[0] - xb) < np.linalg.norm(p[0] - xa):
        p = p[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    if s[-1] <= 0:
        return p
    n = max(int(round(s[-1] / step_px)) + 1, len(p), 5)
    sn = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(sn, s, p[:, 0]), np.interp(sn, s, p[:, 1])])


def _flow_quality(vfield, cl, R):
    """Geometric reliability of a measurement from the time-mean OF field
    (stored per measurement as ``quality``; see scripts/centerline_alignment.py):
      align    speed-weighted |cos θ| between flow vector and CL tangent, |ρ| ≤ R
      consist  speed-weighted fraction of axial flow in the dominant direction
      centre   median over slices of |ρ_peak| / R
    """
    from .centerline import _centerline_geometry
    vf = np.asarray(vfield); cl = np.asarray(cl, float)
    nan = dict(align=float("nan"), consist=float("nan"), centre=float("nan"))
    if vf.ndim != 3 or len(cl) < 3 or not np.isfinite(R) or R <= 0:
        return nan
    H, W = vf.shape[:2]
    _, tang, norm = _centerline_geometry(cl)
    rho = np.linspace(-R, R, 21); ax_all, sp_all, peaks = [], [], []
    for i in range(len(cl)):
        rr = cl[i, 0] + rho * norm[i, 0]; cc = cl[i, 1] + rho * norm[i, 1]
        ok = (rr >= 0) & (rr < H - 1) & (cc >= 0) & (cc < W - 1)
        if ok.sum() < 5:
            continue
        r0 = rr[ok].astype(int); c0 = cc[ok].astype(int); fr = rr[ok] - r0; fc = cc[ok] - c0
        def bil(ch):
            a = vf[..., ch]
            return ((1 - fr) * (1 - fc) * a[r0, c0] + fr * (1 - fc) * a[r0 + 1, c0]
                    + (1 - fr) * fc * a[r0, c0 + 1] + fr * fc * a[r0 + 1, c0 + 1])
        vx, vy = bil(0), bil(1)
        axial = vy * tang[i, 0] + vx * tang[i, 1]
        ax_all.append(axial); sp_all.append(np.hypot(vx, vy))
        peaks.append(abs(rho[ok][np.argmax(np.abs(axial))]) / R)
    if not ax_all:
        return nan
    ax = np.concatenate(ax_all); sp = np.concatenate(sp_all); w = sp / max(sp.sum(), 1e-12)
    dom = np.sign(np.sum(ax))
    return dict(align=float(np.sum(w * np.abs(ax) / np.maximum(sp, 1e-12))),
                consist=float(np.sum(np.abs(ax) * (np.sign(ax) == dom)) / max(np.abs(ax).sum(), 1e-12)),
                centre=float(np.median(peaks)))


# ---------------------------------------------------------------------
# Per-vessel analysis worker
# ---------------------------------------------------------------------

def _smooth_path_inplace(pts, sigma=1.5):
    """Gaussian-smooth a polyline's coordinates along arc, keeping the
    ORIGINAL point spacing.  Net3's raw graph paths trace triangle
    centroids at variable spacing (~1–5 px) with high per-point tangent
    jitter (~50°); a small σ (~1.5 points) gives the downstream CL
    refinement a clean starting point without changing the point count
    (which would rescale the refinement's per-point shift budget)."""
    from scipy.ndimage import gaussian_filter1d as _gf
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return pts
    return np.column_stack([
        _gf(pts[:, 0], sigma=sigma, mode="nearest"),
        _gf(pts[:, 1], sigma=sigma, mode="nearest"),
    ])


def _analyze_vessel(A, B, path_xy, radius_graph, geom, tile_id, stack,
                     f0, pt, ridge_mask=None, ridge_edt=None,
                     keep_kl_inputs=False, of_overrides=None):
    """Run merged OF on one vessel in one tile.  Returns a clean
    measurement dict or None if the vessel isn't sufficiently in-FOV
    or OF fails.  ``keep_kl_inputs`` attaches ``_kl_inputs`` (time-mean
    flow field, refined CL, R_prior, ridge-mask ROI) so the Kloosterman
    step can be re-run offline with other settings (never stored in graphs)."""
    H, W = stack.shape[1], stack.shape[2]
    tc = graph_to_tile_batch(np.asarray(path_xy), geom, tile_id)
    seg_tc = _largest_inbounds_segment(tc, H, W)
    if len(seg_tc) < 5:
        return None
    # Smooth the raw path in place (keep original point count) so
    # downstream CL refinement starts from a clean centerline without
    # rescaling its per-point shift budget.
    seg_raw_precrisp = seg_tc.copy()   # keep for kink_ratio baseline
    # MINIMAL TRIM — identical to the viewer's fresh-OF _get_chain_coords_local:
    # resample to ~1px uniform, then trim min(3px, 0.15·arc) per end (just step
    # off the junction node).  Replaces the EDT junction-bulge trim, which
    # over-trimmed short/junction-adjacent/boundary vessels to stubs and dropped
    # ~16% of them below the arc/aspect QC gates (tile-12 measurement).  The
    # downstream Kloosterman median-band, per-slice R gates, neighbour rejection
    # and curvature guard reject contaminated END SLICES without discarding the
    # whole vessel — so we keep coverage AND match fresh-OF exactly.  (compute
    # resamples internally too; the pre-smooth is dropped to mirror the viewer.)
    if len(seg_tc) >= 2:
        _s = np.concatenate([[0.0], np.cumsum(np.hypot(
            np.diff(seg_tc[:, 0]), np.diff(seg_tc[:, 1])))])
        _tot = float(_s[-1])
        if _tot >= 2.0:
            _n = int(np.clip(round(_tot) + 1, max(5, len(seg_tc)), 400))
            _sn = np.linspace(0.0, _tot, _n)
            seg_tc = np.column_stack([np.interp(_sn, _s, seg_tc[:, 0]),
                                       np.interp(_sn, _s, seg_tc[:, 1])])
    if len(seg_tc) > 4:
        _s = np.concatenate([[0.0], np.cumsum(np.hypot(
            np.diff(seg_tc[:, 0]), np.diff(seg_tc[:, 1])))])
        _tot = float(_s[-1])
        _m = min(3.0, 0.15 * _tot)
        _keep = (_s >= _m) & (_s <= _tot - _m)
        if _keep.sum() >= 5:
            seg_tc = seg_tc[_keep]
    if len(seg_tc) < 5:
        return None
    arc = float(np.hypot(np.diff(seg_tc[:, 0]),
                          np.diff(seg_tc[:, 1])).sum())
    if arc < MIN_ARC_PX:
        return None
    # legacy of_profile expects (row, col) with row = H - y_cartesian.
    # graph_to_tile_batch returns image-frame (row, col) already.
    cl_rc = seg_tc.astype(float)
    # Vessel angle in the tile frame (for radius rescale)
    dr = seg_tc[-1, 0] - seg_tc[0, 0]
    dc = seg_tc[-1, 1] - seg_tc[0, 1]
    # In (row, col) frame, "angle" for radius rescale below is atan2(dr, dc)
    angle = float(np.arctan2(dr, dc))
    R_px = float(graph_radius_to_tile_px(
        float(radius_graph or 5.0), geom, tile_id))
    try:
        u = pt["compute_of_profile_and_Q"](
            stack, cl_rc, R_px, f0_hz=f0,
            frame_dt_s=pt["FRAME_DT_S"],
            ridge_mask=ridge_mask,
            **{**FB_KWARGS, **(of_overrides or {})})
        seg = pt["_rebin_profile"](u, cl_rc, R_px,
                                    n_bins=30, fit_fraction=0.75)
    except Exception:
        return None
    conv = pt["PX_SIZE_UM"] ** 3 / pt["FRAME_DT_S"] * 1e-6
    Q_seg = float(seg.get("Q_pois_mean", np.nan)) * conv
    # Direct-integration Q (plug-flow style): sum(v_axial) * pixel_area
    # over the vessel-mask cross-section.  Robust to non-Poiseuille
    # profiles (bridges, shear-flow, asymmetric branches) where the
    # Poiseuille parabola fails.  Uses pertile's Q_plug_mean.
    Q_direct = float(u.get("Q_plug_mean", np.nan)) * conv
    # Physically-consistent Poiseuille Q from JOINT free fit of (v0, R):
    #   Q_pois_free = v0_free × π R_free² / 2
    # This is what "Poiseuille" actually means in physics — both v0 and
    # R jointly fit to the observed profile.  The stored Q_nls uses
    # v0_fixed at R=R_graph, which systematically under-estimates when
    # the true vessel R > mask R.
    v0_free    = float(u.get("v0_free", np.nan))
    R_free_px  = float(u.get("R_fit_free", np.nan))
    if np.isfinite(v0_free) and np.isfinite(R_free_px) and R_free_px > 0:
        Q_pois_free = abs(v0_free) * np.pi * (R_free_px ** 2) / 2.0 * conv
    else:
        Q_pois_free = float("nan")
    # Kloosterman/Poelma scalar Q — per-cross-section Poiseuille fits,
    # IQR-filtered, averaged.  Robust to branch/junction contamination.
    Q_kl_raw = float(u.get("Q_kloosterman", np.nan))
    Q_kl = Q_kl_raw * conv if np.isfinite(Q_kl_raw) else float("nan")
    R_kl_um = float(u.get("R_kloosterman", np.nan)) * pt["PX_SIZE_UM"]
    V_CL_kl = float(u.get("V_CL_kloosterman", np.nan)) * pt["PX_SIZE_UM"] / pt["FRAME_DT_S"]
    Q_kl_valid_frac = float(u.get("Q_kloosterman_valid_frac", 0.0))
    Q_kl_n_valid = int(u.get("Q_kloosterman_n_valid", 0))
    Qt_ridge = np.asarray(u["Q_pois_t"]) * conv
    ridge_mean = float(np.nanmean(Qt_ridge))
    # Use Q_kl (per-slice Poiseuille, R_fit for area, median across slices
    # — Kloosterman 2014 convention) as the scale target.  Chosen after
    # the 53-tile KCL sweep: Q_kl is citable (published method), robust to
    # per-vessel R_fit blow-ups (per-slice median gates outliers), and
    # kept its KCL residual essentially unchanged when the QC pass rate
    # went from 47% → 58% (Q_pois_free worsened by 2.5 pp on the same
    # sweep).  When Kloosterman couldn't fit enough slices, fall back to the
    # BOUNDED mask-R Poiseuille (Q_seg) FIRST, and only then to the free-R fit
    # (Q_pois_free) — free-R is railing-prone (can blow up on short/blunt
    # vessels), so it must never be preferred over the mask-R estimate.
    if np.isfinite(Q_kl):
        Q_scale = Q_kl; best_key = "kl"
    elif np.isfinite(Q_seg):
        Q_scale = Q_seg; best_key = "pois_seg"
    elif np.isfinite(Q_pois_free):
        Q_scale = Q_pois_free; best_key = "pois_free"
    else:
        Q_scale = Q_seg; best_key = "pois_seg"
    if not (np.isfinite(Q_scale) and np.isfinite(ridge_mean)) \
            or abs(ridge_mean) < 1e-12:
        return None
    Qt = Qt_ridge * (abs(Q_scale) / abs(ridge_mean))
    hr = pt["fit_harmonics"](Qt, pt["FRAME_DT_S"], f0,
                              K=pt["N_HARMONICS"], loss="huber",
                              include_dc=True)
    mean_Q = float(hr.get("a0", Q_scale))
    # Physical direction: sign of mean flow along the arc
    if mean_Q >= 0:
        flow_from, flow_to = A, B
    else:
        flow_from, flow_to = B, A
    # QC on the axial profile — filters blending artifacts / off-CL /
    # asymmetric-branch failures.  R_edt from graph radius as the
    # reference for the peak-offset gate.
    R_edt_um = R_px * pt["PX_SIZE_UM"]
    # CL-geometry gates: refined_kink flags refinement-noise (paired A/V
    # bundles, junction pockets); aspect ratio flags pancake-shaped
    # "vessels" that are wider than they are long.
    # Baseline kink = the raw graph path (pre-smoothing).  Comparing
    # against the smoothed cl_rc would trivially fail the kink_ratio
    # gate since we smoothed away the jitter that kink_ratio measures.
    init_kink = _kink_deg(seg_raw_precrisp)
    cl_refined = u.get("cl_used")
    refined_kink = (_kink_deg(np.asarray(cl_refined))
                     if cl_refined is not None and len(cl_refined) >= 3
                     else float("nan"))
    aspect = arc / max(2.0 * R_px, 1e-6)
    qc = _profile_qc(np.asarray(u["bin_centers"]),
                       np.asarray(u["profile_v"]),
                       R_edt_um, pt["PX_SIZE_UM"], pt["FRAME_DT_S"],
                       init_kink_deg=init_kink,
                       refined_kink_deg=refined_kink,
                       aspect_ratio=aspect)
    out = {
        "schema_version": 4,
        "tile_id": tile_id,
        "flow_from": flow_from, "flow_to": flow_to,
        "harm": {"mean_Q": abs(mean_Q), "f0_hz": f0,
                  "harmonics": hr.get("harmonics", []),
                  "fit_success": True},
        "snr": {"dc_db": float(hr.get("snr_dc_fit_db", np.nan)),
                 "h1_db": float(hr.get("snr_ac_fit_db", np.nan))},
        "qc": qc,
        "geom": {"R_um": R_edt_um,
                  "L_um": arc * pt["PX_SIZE_UM"]},
        "Q_nls": {"best": abs(mean_Q), "best_key": best_key},
        "Q_direct": (float(abs(Q_direct)) if np.isfinite(Q_direct)
                      else float("nan")),
        "Q_pois_free": (float(abs(Q_pois_free)) if np.isfinite(Q_pois_free)
                         else float("nan")),
        "Q_kl": (float(abs(Q_kl)) if np.isfinite(Q_kl) else float("nan")),
        # Alternative scale estimators, kept for estimator comparison
        # (scripts/fresh_tile_measurements.py): mask-R Poiseuille on the
        # segmentation CL, and the raw ridge-CL Poiseuille mean (signed A→B).
        "Q_pois_seg": (float(abs(Q_seg)) if np.isfinite(Q_seg) else float("nan")),
        "Q_ridge_mean": float(ridge_mean),
        "R_kl_um": R_kl_um,
        "V_CL_kl_ums": V_CL_kl,
        "Q_kl_valid_frac": Q_kl_valid_frac,
        "Q_kl_n_valid": Q_kl_n_valid,
        "Q_t": Qt.astype(np.float32),
        "prov": {"method": "plexusflow_vessel_merged_of",
                  "n_cross_sections": int(len(cl_rc)),
                  "R_free_um": float(u.get("R_fit_free", np.nan))
                                * pt["PX_SIZE_UM"],
                  "cl_shift_px": float(u.get("cl_shift", 0.0)),
                  "of_dt": int(u.get("of_dt", FB_KWARGS["dt"])),
                  "of_pilot_disp_px": float(u.get("of_pilot_disp_px", np.nan))},
    }
    # Per-measurement quality record — enough to recompute the reliability
    # scale c (f, alignment, tile) after the batch without re-running OF.
    _psq = np.asarray(u.get("Q_kloosterman_per_slice", []), float)
    _psr = np.asarray(u.get("R_kloosterman_per_slice", []), float)
    _fin = np.isfinite(_psq)
    out["quality"] = dict(
        kl_ok=bool(np.isfinite(Q_kl)),
        n_slices=int(_psq.size), n_kept=int(Q_kl_n_valid), f=float(Q_kl_valid_frac) if np.isfinite(Q_kl) else 0.0,
        slice_cv=(float(np.nanstd(np.abs(_psq[_fin])) / max(np.nanmean(np.abs(_psq[_fin])), 1e-12))
                  if _fin.sum() >= 2 else float("nan")),
        R_ratio_med=(float(np.nanmedian(_psr[np.isfinite(_psr)]) / R_px) if np.isfinite(_psr).any() else float("nan")),
        **_flow_quality(u.get("vfield_mean"), u.get("cl_used"), float(u.get("R_used", R_px))))
    if keep_kl_inputs:
        vf = np.asarray(u["vfield_mean"], np.float32)
        r0, c0 = u.get("roi_offset", (0, 0))
        rm = (np.asarray(ridge_mask[r0:r0 + vf.shape[0], c0:c0 + vf.shape[1]], bool)
              if ridge_mask is not None else None)
        out["_kl_inputs"] = dict(vfield_mean=vf, cl_used=np.asarray(u["cl_used"], float),
                                 R_prior_px=float(u.get("R_used", R_px)), roi_offset=(r0, c0),
                                 ridge_mask_roi=rm, conv=float(conv))
    return out


# ---------------------------------------------------------------------
# Public entry: process the stage
# ---------------------------------------------------------------------

def _tile_edge_midpoints(G: nx.Graph) -> dict:
    """Return {tile_id: (median_x, median_y)} over edges that tile measured."""
    from collections import defaultdict
    per = defaultdict(list)
    for u, v, d in G.edges(data=True):
        xu, yu = G.nodes[u].get("x"), G.nodes[u].get("y")
        xv, yv = G.nodes[v].get("x"), G.nodes[v].get("y")
        if xu is None or xv is None: continue
        mx, my = 0.5 * (xu + xv), 0.5 * (yu + yv)
        for m in (d.get("measurements_piv") or []):
            t = m.get("tile_id")
            if t is not None:
                per[int(t)].append((mx, my))
    return {t: (float(np.median([p[0] for p in pts])),
                float(np.median([p[1] for p in pts])))
            for t, pts in per.items() if pts}


def _tile_spatial_neighbors(G: nx.Graph, k: int = 6) -> dict:
    """Return {tile_id: [nearest-k spatial-neighbor tile_ids]}."""
    centers = _tile_edge_midpoints(G)
    ids = sorted(centers)
    if len(ids) < 2:
        return {t: [] for t in ids}
    xy = np.array([centers[t] for t in ids])
    out = {}
    for i, t in enumerate(ids):
        d2 = np.sum((xy - xy[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argsort(d2)[: min(k, len(ids) - 1)]
        out[t] = [ids[j] for j in nn]
    return out


def _tile_qt_spectrum_peak(G: nx.Graph, tile: int, fs: float = 250.0,
                            f_lo: float = 0.8, f_hi: float = 3.5,
                            n_top: int = 20) -> Optional[float]:
    """Tile heart rate from its own traces: stacked, peak-normalised Hann
    periodograms of the ``n_top`` strongest Q(t) traces (by |mean Q|), most
    prominent peak in [f_lo, f_hi] — the same estimator the batch uses for f0.
    (Averaging |FFT| over ALL traces, as before 2026-10-07, let the 1/f noise of
    the many weak traces win: tile 53 peaked at 0.73 Hz instead of 3.02.)"""
    from .harmonic import estimate_f0_in_band, consensus_f0_stacked
    cand = []
    for _u, _v, d in G.edges(data=True):
        for m in (d.get("measurements_piv") or []):
            if m.get("tile_id") != tile:
                continue
            q = m.get("Q_t"); mq = (m.get("harm") or {}).get("mean_Q")
            if q is not None and len(q) > 64 and mq is not None and np.isfinite(mq):
                q = np.asarray(q, float); q = q[np.isfinite(q)]
                if q.size > 64:     # linear detrend: with few beats, drift leaks into the low-f band
                    tt = np.arange(q.size); q = q - np.polyval(np.polyfit(tt, q, 1), tt)
                cand.append((abs(float(mq)), q))
    cand.sort(key=lambda x: -x[0])
    if cand:                                   # ≥ 1 full beat per recording: floor the search at 1/T
        f_lo = max(f_lo, fs / float(np.median([len(q) for _, q in cand[:n_top]])))
    pgs = []
    for _, q in cand[:n_top]:
        q = q[np.isfinite(q)]
        if q.size < 64:
            continue
        f_est, fr, pw = estimate_f0_in_band(q, 1.0 / fs, fmin_hz=f_lo, fmax_hz=f_hi, return_periodogram=True)
        if np.isfinite(f_est):
            pgs.append((fr, pw))
    if not pgs:
        return None
    f0 = consensus_f0_stacked(pgs, fmin_hz=f_lo, fmax_hz=f_hi)
    return float(f0) if np.isfinite(f0) else None


def _refit_tile_harm(G: nx.Graph, tile: int, f0_hz: float,
                      fs: float = 250.0, n_h: int = 3) -> int:
    """Refit each of the tile's measurements' ``harm`` at a corrected f0 with
    the batch harmonic fitter (Huber IRLS), keeping the batch field layout
    (mean_Q, f0_hz, harmonics, fit_success, sigma_DC, sigma_resid, dc_snr_db)
    and ``snr``.  Returns the number of measurements touched."""
    from .harmonic import fit_harmonics
    n = 0
    for _u, _v, d in G.edges(data=True):
        for m in (d.get("measurements_piv") or []):
            if m.get("tile_id") != tile or m.get("Q_t") is None:
                continue
            Qt = np.asarray(m["Q_t"], dtype=float)
            if len(Qt) < 32:
                continue
            hr = fit_harmonics(Qt, 1.0 / fs, f0_hz, K=n_h, loss="huber", include_dc=True)
            a0, s0 = float(hr["a0"]), float(hr["sigma_a0"])
            resid = np.asarray(hr["resid"], float); resid = resid[np.isfinite(resid)]
            dc_db = float(20 * np.log10(abs(a0) / s0)) if s0 > 0 and a0 != 0 else float("nan")
            m["harm"] = dict(mean_Q=abs(a0), f0_hz=float(f0_hz), harmonics=hr["harmonics"], fit_success=True,
                             sigma_DC=s0, sigma_resid=float(resid.std(ddof=2 * n_h + 1)) if resid.size > 2 * n_h + 1 else float("nan"),
                             dc_snr_db=dc_db)
            h1 = next((h for h in hr["harmonics"] if h["k"] == 1), {})
            m["snr"] = dict(m.get("snr") or {}, dc_db=dc_db, h1_db=float(h1.get("snr_db", np.nan)))
            n += 1
    return n


def _tile_beat_r2(G: nx.Graph, tile: int, f0: float, fs: float = 250.0, n_top: int = 20) -> float:
    """Beat strength: median R² of a DC + H1 + H2 fit at f0 over the tile's n_top strongest Kloosterman-valid traces."""
    from .harmonic import fit_harmonics
    ms = sorted([m for *_, d in G.edges(data=True) for m in (d.get("measurements_piv") or [])
                 if m.get("tile_id") == tile and (m.get("quality") or {}).get("kl_ok") and m.get("Q_t") is not None],
                key=lambda m: -abs((m.get("harm") or {}).get("mean_Q") or 0))[:n_top]
    r = []
    for m in ms:
        q = np.asarray(m["Q_t"], float); q = q[np.isfinite(q)]
        if q.size < 50 or q.var() == 0:
            continue
        hr = fit_harmonics(q, 1.0 / fs, f0, K=2, loss="huber", include_dc=True)
        res = np.asarray(hr["resid"], float); res = res[np.isfinite(res)]
        r.append(1 - res.var() / q.var())
    return float(np.median(r)) if r else float("nan")


def _sanity_check_tile_f0(G: nx.Graph, self_tol: float = 0.05,
                           agree_frac: float = 0.15,
                           verbose: bool = True, refit: bool = True,
                           f_lo: float = 0.8, f_hi: float = 3.5,
                           r2_min: float = 0.3, r2_gain: float = 0.05) -> int:
    """Check each tile's stored f0 against its OWN traces and replace it when the evidence says so.

    Estimate (benchmarked 2026-10-14 against user-reviewed 15S rates and clear 21S/27S tiles: 98/102 correct, no octave
    errors): stacked periodograms of the tile's strongest traces, linear-detrended, search floor max(f_lo, 1/T) so every
    recording holds ≥ 1 full beat (`_tile_qt_spectrum_peak`).

    Acceptance is by the tile's own data, not its neighbours (heart rate varies between tiles legitimately — warm-up,
    the 27S dip, the wandering 15S rate): replace the stored f0 when the new rate gives a clear beat (median R² of a
    DC+H1+H2 fit ≥ ``r2_min``) that beats the stored rate's by ≥ ``r2_gain``.  Tiles reviewed by hand
    (graph['tile_f0_status'][t]['user_checked']) are never changed.  Returns the number of tiles refit."""
    tf = G.graph.get("tile_f0_piv") or {}
    if not tf:
        return 0
    status = G.graph.get("tile_f0_status") or {}
    orig = dict(tf); changes = {}
    for t, f0 in orig.items():
        if (status.get(t) or {}).get("user_checked"):
            continue
        peak = _tile_qt_spectrum_peak(G, t, f_lo=f_lo, f_hi=f_hi)
        if peak is None or not np.isfinite(peak) or abs(float(f0) - peak) / peak <= self_tol:
            continue
        r_old, r_new = _tile_beat_r2(G, t, float(f0)), _tile_beat_r2(G, t, peak)
        if np.isfinite(r_new) and r_new >= r2_min and (not np.isfinite(r_old) or r_new >= r_old + r2_gain):
            changes[t] = peak
            if verbose:
                print(f"[sanity] tile {t}: f0 {float(f0):.3f} → {peak:.3f} Hz (own traces; beat R² {r_old:.2f} → {r_new:.2f})")
        elif verbose:
            print(f"[sanity] tile {t}: stored f0 {float(f0):.3f} vs own-trace estimate {peak:.3f} Hz — kept "
                  f"(beat R² {r_old:.2f} vs {r_new:.2f}; not a clear improvement)")
    for t, f in changes.items():
        tf[t] = float(f)
        if refit:
            n = _refit_tile_harm(G, t, f)
            if verbose:
                print(f"[sanity]   refit {n} measurements on tile {t}")
    G.graph["tile_f0_piv"] = tf
    return len(changes)

def analyze_stage(cfg: StageConfig,
                    tile_ids: Optional[Iterable[int]] = None,
                    max_frames: Optional[int] = None,
                    n_workers: Optional[int] = None,
                    default_f0_hz: float = 2.6,
                    skip_analyzed: bool = False,
                    refit_f0: bool = False,
                    dry_run: bool = False,
                    verbose: bool = True,
                    kl_sidecar_dir: Optional[Path] = None,
                    out_path: Optional[Path] = None) -> Path:
    """Run the batch analyzer.

    Reads: ``cfg.graph_vessels_path`` (make with `plexusflow condense`),
    ``cfg.positions_path``, per-tile raw videos.

    Writes: measurements into edge ``measurements_piv`` and saves the
    graph incrementally after every tile to ``cfg.graph_piv_path``
    (or ``cfg.graph_vessels_path`` if that already carries measurements
    — supports resume).
    """
    pt = _import_pertile()
    # Heart-rate search band and OF frame spacing come from the stage config (plexusflow.yaml).
    # Default band widened 2026-08-24 from [1.2, 4.5] (early, cold tiles ~0.9–1.1 Hz locked onto H2 as fundamental).
    F0_FMIN_HZ = float(cfg.f0_min_hz)     # default 0.8; 15S (heart just started, ~0.6–1.2 Hz) uses 0.5–1.6
    F0_FMAX_HZ = float(cfg.f0_max_hz)     # default 3.5
    def _of_spec(spec):
        if str(spec).lower() == "auto":
            return dict(dt="auto", dt_max=int(cfg.of_dt_max), target_disp_px=float(cfg.of_target_disp_px))
        return dict(dt=int(spec))
    of_overrides = _of_spec(cfg.of_dt)
    _by_tile = {int(k): v for k, v in (cfg.of_dt_by_tile or {}).items()}

    def _of_for(tid):
        """Per-tile OF frame spacing: of_dt_by_tile[tid] if set, else the stage-wide of_dt."""
        return _of_spec(_by_tile[int(tid)]) if int(tid) in _by_tile else of_overrides

    src = cfg.graph_vessels_path
    if not src.exists():
        raise FileNotFoundError(
            f'no vessel graph at {src}. Run `plexusflow condense` first.')
    out = Path(out_path) if out_path is not None else cfg.graph_piv_path
    # Resume: prefer output graph if it already has data.  With a custom
    # out_path that doesn't exist yet, start from the stage's measured graph
    # (not the unmeasured graph_vessels), so edits/attributes carry over.
    if out.exists():
        load_path = out
    elif out_path is not None and cfg.graph_piv_path.exists():
        load_path = cfg.graph_piv_path
    else:
        load_path = src
    if verbose:
        print(f'loading {load_path.name}')
    G = pickle.loads(load_path.read_bytes())
    n_pre = sum(1 for _, _, d in G.edges(data=True)
                 if d.get("measurements_piv"))
    if verbose:
        print(f'  {G.number_of_nodes()} nodes, '
              f'{G.number_of_edges()} vessels, '
              f'{n_pre} vessels already with PIV')

    # Discover tile IDs
    all_tile_ids = cfg.discover_raw_tile_ids()
    if tile_ids is None:
        tile_ids = list(all_tile_ids)
    tile_ids = sorted(set(int(t) for t in tile_ids))
    # Tiles excluded from flow measurement (stage config ``exclude_tiles``, e.g. no detectable flow): never measured,
    # and any measurements already stored from them are dropped so those vessels count as unimaged downstream.
    _excl = {int(t) for t in (cfg.exclude_tiles or [])}
    if _excl:
        tile_ids = [t for t in tile_ids if t not in _excl]
        _n_drop = 0
        for _a, _b, _d in G.edges(data=True):
            if _d.get("measurements_piv"):
                _keep = [mm for mm in _d["measurements_piv"] if int(mm.get("tile_id", -1)) not in _excl]
                _n_drop += len(_d["measurements_piv"]) - len(_keep); _d["measurements_piv"] = _keep
        if verbose:
            print(f'  exclude_tiles {sorted(_excl)}: not measured; dropped {_n_drop} stored measurements')

    # Peek shapes for EVERY discoverable tile — the mosaic bbox must
    # be computed from all tiles, not just the subset we'll process
    # (else the canvas comes out smaller than fuse actually wrote).
    if verbose:
        print(f'peeking video shapes for {len(all_tile_ids)} tiles ...')
    all_shapes: dict[int, tuple[int, int]] = {}
    for tid in all_tile_ids:
        shp = tile_video_shape(cfg, tid)
        if shp is not None:
            all_shapes[tid] = shp
    if not all_shapes:
        raise RuntimeError('no raw videos found; '
                           'check external_raw_video_dir in plexusflow.yaml')
    shapes = {tid: all_shapes[tid] for tid in tile_ids
              if tid in all_shapes}
    geom = load_tile_geometry(cfg, video_shapes=all_shapes)

    # Optional: load the fuse-frame mask once so we can warp it per-tile
    # into the OF ridge-refinement gate.  If mask.tif isn't present,
    # refinement runs without avascular-noise suppression.
    fuse_mask = None
    if cfg.mask_path.exists():
        _mask_arr = tifffile.imread(cfg.mask_path)
        fuse_mask = (_mask_arr > 0)
        if verbose:
            print(f'  loaded mask {cfg.mask_path.name} for OF ridge gating '
                  f'({fuse_mask.sum():,} true px)')
    if verbose:
        print(f'  mosaic geom: fuse {geom.fuse_shape}, '
              f'ds={geom.downsample}, bbox origin=({geom.y0:.0f}, {geom.x0:.0f})')

    # Per-tile f0 (persisted on the graph so re-runs skip re-estimation).
    if G.graph.get("tile_f0_piv") is None:
        G.graph["tile_f0_piv"] = {}
    f0_by_tile = G.graph["tile_f0_piv"]
    # Pre-pass: correct cached per-tile f0 from the existing traces BEFORE
    # measuring — OF uses f0 to pick systolic frames for CL refinement and the
    # flow tangent, so a wrong f0 must not reach the new measurements.
    if f0_by_tile and not skip_analyzed:
        try:
            _n0 = _sanity_check_tile_f0(G, verbose=verbose, refit=False, f_lo=F0_FMIN_HZ, f_hi=F0_FMAX_HZ)
            f0_by_tile = G.graph["tile_f0_piv"]
            if verbose and _n0:
                print(f"[sanity pre-pass] corrected cached f0 on {_n0} tile(s) before measuring")
        except Exception as _e:
            print(f"[sanity pre-pass] skipped: {_e}")
    def _f0(tid):
        v = f0_by_tile.get(tid, f0_by_tile.get(str(tid)))
        return float(v) if v is not None and np.isfinite(float(v)) \
            else float(default_f0_hz)

    def _estimate_tile_f0(stack: np.ndarray, cand_edges: list,
                           geom_, tid_, ridge_mask_, ridge_edt_,
                           sample_n: int = 12) -> float:
        """Two-pass f0 estimation, matching pertile's approach:
          1. run compute_of_profile_and_Q on the ``sample_n`` longest
             candidate vessels with a dummy f0 → get their Q(t) traces
          2. estimate consensus f0 from those periodograms
        Falls back to ``default_f0_hz`` if no vessel Q(t) can be fit.
        """
        from .harmonic import (estimate_f0_in_band,
                                 consensus_f0_stacked)
        from .constants import FRAME_DT_S as _dt
        # Pick the longest ``sample_n`` candidates.
        def _plen(e):
            p = e[2].get("path")
            try:
                return int(np.asarray(p).shape[0]) if p is not None else 0
            except Exception:
                return 0
        cand_ranked = sorted(cand_edges, key=_plen, reverse=True)[:sample_n]
        periodograms = []
        vessel_f0s = []
        for A, B, d in cand_ranked:
            try:
                m = _analyze_vessel(
                    A, B, _path_a_to_b(G, A, B, d["path"]),
                    d.get("radius"),
                    geom_, tid_, stack, float(default_f0_hz), pt,
                    ridge_mask=ridge_mask_, ridge_edt=ridge_edt_,
                    of_overrides=_of_for(tid_))
            except Exception:
                continue
            if m is None:
                continue
            Qt = m.get("Q_t")
            if Qt is None:
                continue
            Qt = np.asarray(Qt, float)
            finite = Qt[np.isfinite(Qt)]
            if finite.size < 20 or np.nanstd(finite) < 1e-12:
                continue
            try:
                f_est, freqs, power = estimate_f0_in_band(
                    finite, _dt,
                    fmin_hz=F0_FMIN_HZ, fmax_hz=F0_FMAX_HZ,
                    return_periodogram=True)
                if np.isfinite(f_est):
                    periodograms.append((freqs, power))
                    vessel_f0s.append(f_est)
            except Exception:
                continue
        if periodograms:
            f0 = consensus_f0_stacked(periodograms,
                                        fmin_hz=F0_FMIN_HZ, fmax_hz=F0_FMAX_HZ)
            if np.isfinite(f0):
                return float(f0)
            if vessel_f0s:
                return float(np.median(vessel_f0s))
        return float(default_f0_hz)

    # Parallelism (joblib+loky over vessels within each tile).  Set cv2
    # threads to 0 so we get real process-level parallelism.
    import cv2 as _cv2
    _cv2.setNumThreads(0)
    try:
        from joblib import Parallel, delayed, cpu_count
    except ImportError:
        Parallel = None
        def cpu_count(): return 1
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    load_tiff_stack = pt["load_tiff_stack"]
    cut_before_fade = pt["cut_before_fade"]

    total_measured = 0
    _t_run = time.time()
    for tid in tile_ids:
        if tid not in shapes:
            print(f'tile {tid}: no video, skip'); continue
        t0 = time.time()
        try:
            folder = cfg.raw_tile_folder(tid)
        except FileNotFoundError:
            print(f'tile {tid}: missing folder, skip'); continue
        # Frame budget = the viewer's fresh-OF: load up to max_frames then
        # cut_before_fade trims the fade-out, leaving the full usable video
        # (~6–9 cardiac cycles at max_frames=900).  Matches fresh-OF exactly.
        stack = load_tiff_stack(str(folder), max_frames=max_frames)
        stack, _ = cut_before_fade(stack, verbose=False)
        stack = np.asarray(stack, dtype=np.float32)
        Hh, Ww = stack.shape[1], stack.shape[2]

        # Pre-filter vessels to those with >= MIN_ARC_PX arc inside this tile
        cand_edges = []
        for A, B, d in G.edges(data=True):
            path = d.get("path")
            if path is None:
                continue
            path_arr = np.asarray(path, dtype=float)
            if len(path_arr) < 2:
                continue
            if skip_analyzed:
                already = any(m.get("tile_id") == tid
                              for m in d.get("measurements_piv", []))
                if already:
                    continue
            # resampled + oriented path: short 2–4-point paths (≈1850 edges,
            # incl. node-moved V stubs) otherwise fail the ≥5-point in-FOV test
            if _in_fov_arc(_path_a_to_b(G, A, B, path_arr), geom, tid, Hh, Ww) >= MIN_ARC_PX:
                cand_edges.append((A, B, d))

        if not cand_edges:
            print(f'tile {tid}: 0 vessels in FOV  '
                  f'(video {stack.shape[0]}f  '
                  f'{time.time()-t0:.0f}s)', flush=True)
            continue

        ridge_mask_tile = (warp_mask_to_tile(fuse_mask, geom, tid)
                            if fuse_mask is not None else None)
        if ridge_mask_tile is not None:
            from scipy.ndimage import distance_transform_edt as _edt
            ridge_edt_tile = _edt(ridge_mask_tile)
        else:
            ridge_edt_tile = None

        # Estimate + cache per-tile f0 from Q(t) of a sample of vessels.
        # Skips when the graph already carries a valid f0 (resume-safe);
        # refits when refit_f0=True.
        cached_f0 = f0_by_tile.get(tid, f0_by_tile.get(str(tid)))
        if refit_f0 or cached_f0 is None or not np.isfinite(float(cached_f0)):
            f0_est = _estimate_tile_f0(
                stack, cand_edges, geom, tid,
                ridge_mask_tile, ridge_edt_tile)
            f0_by_tile[tid] = f0_est
            print(f'tile {tid}: estimated f0 = {f0_est:.3f} Hz '
                  f'(consensus over sample Q(t))', flush=True)
        f0 = _f0(tid)

        # Workers get ONLY (A, B, oriented path, radius).  A closure that
        # references G (240 MB) or the edge dict (all stored measurements)
        # is re-serialised for every task: profiled 10× slower (2026-10-07).
        _keep = kl_sidecar_dir is not None
        _jobs = [(A, B, _path_a_to_b(G, A, B, d["path"]), d.get("radius"))
                 for A, B, d in cand_edges]

        def _work(A, B, path_xy, radius, _rm=ridge_mask_tile, _re=ridge_edt_tile, _ov=_of_for(tid)):
            return _analyze_vessel(
                A, B, path_xy, radius,
                geom, tid, stack, f0, pt,
                ridge_mask=_rm, ridge_edt=_re,
                keep_kl_inputs=_keep, of_overrides=_ov)

        if Parallel is not None and n_workers > 1:
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_work)(*j) for j in _jobs)
        else:
            results = [_work(*j) for j in _jobs]

        n_ves = 0
        if not skip_analyzed:
            # full re-run of this tile: drop ALL its old measurements first, so a
            # vessel that is no longer measurable doesn't keep a stale value
            for _a, _b, _d in G.edges(data=True):
                if _d.get("measurements_piv"):
                    _d["measurements_piv"] = [mm for mm in _d["measurements_piv"]
                                              if mm.get("tile_id") != tid]
        sidecar = {}
        for (A, B, d), m in zip(cand_edges, results):
            if m is None:
                continue
            ki = m.pop("_kl_inputs", None)
            if ki is not None:
                ki["vfield_mean"] = ki["vfield_mean"].astype(np.float16)
                sidecar[(A, B)] = ki
            # replace any existing measurement for this tile (idempotent)
            d["measurements_piv"] = [mm for mm in d.get("measurements_piv", [])
                                       if mm.get("tile_id") != tid]
            d["measurements_piv"].append(m)
            n_ves += 1
        if kl_sidecar_dir is not None and sidecar and not dry_run:
            import gzip
            Path(kl_sidecar_dir).mkdir(parents=True, exist_ok=True)
            sc_path = Path(kl_sidecar_dir) / f"tile_{tid}.pkl.gz"
            if skip_analyzed and sc_path.exists():
                # incremental run: keep the inputs of vessels measured earlier
                # (only those still in the graph), add the new ones
                with gzip.open(sc_path, "rb") as fh:
                    old = pickle.load(fh).get("kl_inputs", {})
                sidecar = {**{e: v for e, v in old.items() if G.has_edge(*e)}, **sidecar}
            with gzip.open(sc_path, "wb") as fh:
                pickle.dump(dict(tile=tid, f0=f0, kl_inputs=sidecar), fh, protocol=4)
        total_measured += n_ves

        print(f'tile {tid}: {stack.shape[0]}f  f0={f0:.2f}  '
              f'→ {n_ves}/{len(cand_edges)} vessels measured '
              f'({time.time()-t0:.0f}s, {n_workers}w)', flush=True)
        _done = tile_ids.index(tid) + 1; _el = time.time() - _t_run
        print(f'[progress] {_done}/{len(tile_ids)} tiles  elapsed {_el / 60:.1f} min  '
              f'ETA {_el / _done * (len(tile_ids) - _done) / 60:.1f} min', flush=True)

        # incremental save so a crash mid-run keeps progress
        if not dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(pickle.dumps(G))

    if verbose:
        print(f'\ntotal vessel-measurements written this run: {total_measured}')

    # ── Spatial-neighbor sanity check on per-tile f0 ─────────────────
    # Detects tiles whose f0 was picked >30% off their spatial neighbors'
    # median (typical subharmonic mistake), reverifies via that tile's own
    # aggregated Q(t) spectrum, and — if the spectrum peak agrees with
    # neighbors — refits every measurement's `harm` at the corrected f0.
    if not dry_run:
        try:
            _n_fixed = _sanity_check_tile_f0(G, verbose=verbose, f_lo=F0_FMIN_HZ, f_hi=F0_FMAX_HZ)
            if verbose and _n_fixed:
                print(f'[sanity] refit measurements on {_n_fixed} tile(s) '
                      f'whose f0 was inconsistent with spatial neighbors')
        except Exception as _e:
            print(f'[sanity] neighbor f0 sanity check skipped: {_e}')

    if not dry_run:
        cfg.write_provenance("graph_piv", extra={
            "source_graph": src.name,
            "output_graph": out.name,
            "n_new_measurements": total_measured,
            "default_f0_hz": default_f0_hz,
            "max_frames": max_frames,
            "n_workers": n_workers,
        })
        if verbose:
            print(f'saved {out.name}')
    return out
