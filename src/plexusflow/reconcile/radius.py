"""Anatomical vessel radius R(s) from the segmentation, separate from the velocity-profile radius.

Two radii exist in the measurement and they mean different things:

* the **anatomical radius** R_k(s) — where the lumen wall is.  Taken from the segmentation (the stage mask net3 vectorised,
  ``<stage>/mask/mask.tif``): floating-point Euclidean distance transform sampled bilinearly along the vessel centreline,
  middle 70 % of the arc (junction bulges excluded), mask boundary = EDT − 0.5 px (the EDT at a pixel centre is the distance
  to the nearest background pixel centre).  The mask sits inside the true lumen; a single additive offset Δ (px) maps the
  mask boundary onto the lumen wall.  Δ comes from an independent boundary calibration (``calibrate_offset``), not from the
  flow data.  Used for the passive (r⁴/ℓ) and pulsatile network models.
* the **effective profile radius** R_i,fit — the curvature of the measured velocity profile in Kloosterman's section fits.
  Depth averaging and optical-flow smoothing widen it, so it is a nuisance parameter of the flux estimator and is never used
  as geometry.

Per vessel this returns the median radius — the value the network models use — plus diagnostics: the conduction-equivalent
radius ((mean r⁻⁴)^(−1/4), the uniform tube with the same Poiseuille resistance), the taper ratio (conduction / median) and the
detrended along-vessel scatter.  The conduction-equivalent radius is NOT used: r⁻⁴ is dominated by local pinches of the mask
(segmentation artefacts), and with it the passive r⁴/ℓ model matches the reconciled flows far worse (21S corr 0.68 vs 0.78
with the median; the stored net3 radius gives 0.80).  The stored net3 ``radius`` (integer EDT at nodes, junction-inflated) reads about
1 px above the float mask boundary.
"""
from __future__ import annotations

import numpy as np

TRIM = 0.15          # fraction of the arc dropped at each end (junction bulges)


def mask_edt(mask):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(np.asarray(mask) > 0)


def boundary_profile(edt, path_xy, trim=TRIM, search=None, smooth=None):
    """Mask-boundary radius at ~1 px spacing along the middle of a graph path ((x, y), row = H − y).  None if too short.

    The graph path is a polyline between sparse nodes and cuts toward the inner wall of curved vessels, where the EDT reads
    less than the radius; so at each arc position the EDT is maximised along the (smoothed) normal within ±``search`` px
    (default: the EDT at the path point, at least 3 px; 0 = no search) — the medial-axis value, i.e. the largest inscribed
    disc there."""
    from scipy.ndimage import map_coordinates, gaussian_filter1d
    p = np.asarray(path_xy, float)
    if len(p) < 2:
        return None
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]; L = s[-1]
    if L < 4:
        return None
    ss = np.linspace(trim * L, (1 - trim) * L, max(int((1 - 2 * trim) * L), 3))
    rows, cols = edt.shape[0] - np.interp(ss, s, p[:, 1]), np.interp(ss, s, p[:, 0])
    r0 = map_coordinates(edt, [rows, cols], order=1)
    sg = smooth if smooth is not None else max(float(np.median(r0)), 2.0)
    d = np.column_stack([gaussian_filter1d(rows, sg, order=1, mode="nearest"), gaussian_filter1d(cols, sg, order=1, mode="nearest")])
    n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    w = np.maximum(r0, 3.0) if search is None else np.full_like(r0, float(search))     # search=0: EDT on the path itself
    off = np.linspace(-1, 1, 41)[None, :] * w[:, None]
    r = map_coordinates(edt, [(rows[:, None] + off * n[:, :1]).ravel(), (cols[:, None] + off * n[:, 1:]).ravel()], order=1).reshape(off.shape)
    r = r.max(axis=1) - 0.5
    r = r[r > 0]
    return r if len(r) >= 3 else None


def summarize(r, offset_px=0.0):
    """Median, conduction-equivalent radius, taper and detrended scatter of a profile shifted by ``offset_px``."""
    a = np.asarray(r, float) + offset_px
    med = float(np.median(a)); cond = float(np.mean(a ** -4) ** -0.25)
    k = max(3, len(a) // 4); trend = np.convolve(np.pad(a, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(a)]
    return dict(median=med, cond=cond, taper=cond / med, scatter=float(1.4826 * np.median(np.abs(a - trend))))


def calibrate_offset(edt, G, calib_rows, n_boot=2000, seed=0):
    """Additive mask-boundary → lumen offset Δ (px) from independent lumen-width measurements.

    ``calib_rows``: [{edge, R_true}] with R_true the lumen radius in mosaic px (e.g. from matching the temporal-SD band width
    of the real vessel with synthetic vessels of known radius).  Δ = median(R_true − median boundary radius), with a vessel
    bootstrap 95 % interval; the multiplicative alternative (R_true / boundary) is reported for comparison."""
    d, f = [], []
    for row in calib_rows:
        e = tuple(row["edge"]); e = e if G.has_edge(*e) else e[::-1]
        r = boundary_profile(edt, G.edges[e]["path"]) if G.has_edge(*e) else None
        if r is None or not np.isfinite(row["R_true"]):
            continue
        m = float(np.median(r)); d.append(row["R_true"] - m); f.append(row["R_true"] / m)
    d, f = np.array(d), np.array(f); rng = np.random.default_rng(seed)
    bd = np.median(d[rng.integers(0, len(d), (n_boot, len(d)))], axis=1)
    return dict(offset_px=float(np.median(d)), offset_ci=[float(x) for x in np.percentile(bd, [2.5, 97.5])],
                offset_sd=float(np.std(d)), factor=float(np.median(f)), n=int(len(d)))


def anatomical_radii(G, edt, edges, offset_px):
    """{edge: summary} of the calibrated anatomical radius (mask boundary + Δ) for each edge; edges without a usable profile
    fall back to the stored net3 radius − 1 px + Δ (the median net3 − float-boundary difference) and are marked."""
    out = {}
    for e in edges:
        r = boundary_profile(edt, G.edges[e]["path"]) if G.edges[e].get("path") is not None else None
        if r is None:
            rr = max(float(G.edges[e]["radius"]) - 1.0, 0.5) + offset_px
            out[e] = dict(median=rr, cond=rr, taper=1.0, scatter=np.nan, fallback=True)
        else:
            out[e] = dict(**summarize(r, offset_px), fallback=False)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Tube-interior model radius (adopted 2026-11-17).  The network models (passive r⁴/ℓ and the pulsatile two-port) use the
# segmentation radius measured only where the vessel is a tube: at ~1 px spacing along the graph path, the distances L, R
# to the first background pixel on either side along the (smoothed) normal; a point is *walled* when both walls are found
# within CAP × the inscribed radius e (EDT at the point) and the chord is tube-like, L + R ≤ K_CHORD · 2e + 1 px (it fails
# where the normal runs into a junction or a neighbouring lumen).  Half-width a = (L + R)/2 over walled points only:
#     resistance radius  a_med = median a              (robust to mask pinches at junction necks; (mean a⁻⁴)^(−1/4) is not)
#     compliance radius  a_C   = (mean a²)^(1/2)       (volume-equivalent: compliance per length ∝ lumen area)
# Vessels with fewer than MIN_WALLED walled points or a walled fraction below MIN_FRAC keep the stored net3 radius for both
# (flagged ``profiled = False``).  Uncalibrated: no offset or factor is fitted to the flow-profile radius.  The consensus
# radius √(R_graph · R_kl) and the conduction-equivalent radius are kept only as sensitivity analyses.
K_CHORD, CAP, MIN_WALLED, MIN_FRAC = 1.5, 4.0, 5, 0.3


def tube_interior_profile(mask, path_xy, edt=None, k_chord=K_CHORD, cap=CAP):
    """(half-width a, walled) at ~1 px spacing along a graph path ((x, y), row = H − y) on a boolean mask; None if the path
    is shorter than 2 px."""
    from scipy import ndimage as ndi
    mask = np.asarray(mask, bool); H, W = mask.shape
    edt = mask_edt(mask) if edt is None else edt
    p = np.asarray(path_xy, float)
    if len(p) < 2:
        return None
    q = np.column_stack([p[:, 0], H - p[:, 1]])                                  # (col, row)
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    if sl[-1] < 2:
        return None
    si = np.linspace(0, sl[-1], int(np.ceil(sl[-1])) + 1)
    q = np.column_stack([np.interp(si, sl, q[:, 0]), np.interp(si, sl, q[:, 1])])
    s = ndi.gaussian_filter1d(q, 2, axis=0, mode="nearest"); t = np.gradient(s, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12; nrm = np.column_stack([-t[:, 1], t[:, 0]])
    ci, ri = np.clip(q[:, 0].round().astype(int), 0, W - 1), np.clip(q[:, 1].round().astype(int), 0, H - 1)
    e = np.maximum(edt[ri, ci], 0.5); cp = np.maximum(cap * e, 12.0); steps = np.arange(0.5, cp.max() + 0.5, 0.5); out = []
    for sgn in (1, -1):
        X = q[:, None, 0] + sgn * nrm[:, None, 0] * steps; Y = q[:, None, 1] + sgn * nrm[:, None, 1] * steps
        xi, yi = X.round().astype(int), Y.round().astype(int); inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        bg = np.ones_like(inside); bg[inside] = ~mask[yi[inside], xi[inside]]; bg &= steps[None, :] <= cp[:, None] + 1e-9
        out.append(np.where(bg.any(1), steps[bg.argmax(1)], np.inf))
    L, R = out; w = L + R
    walled = np.isfinite(w) & (w <= k_chord * 2 * e + 1.0)
    return (L + R) / 2, walled


def tube_radii(G, edges, mask, min_walled=MIN_WALLED, min_frac=MIN_FRAC):
    """Per edge (mosaic px): resistance radius (median walled half-width), compliance radius (√ mean a²), profiled flag and
    walled fraction.  Unprofiled edges fall back to the stored net3 ``radius`` for both."""
    mask = np.asarray(mask, bool); edt = mask_edt(mask); n = len(edges)
    rg = np.array([float(G.edges[e]["radius"]) for e in edges])
    a_res, a_comp, ok, frac = rg.copy(), rg.copy(), np.zeros(n, bool), np.zeros(n)
    for i, e in enumerate(edges):
        path = G.edges[e].get("path")
        pr = tube_interior_profile(mask, path, edt=edt) if path is not None else None
        if pr is None:
            continue
        a, wl = pr; frac[i] = wl.mean()
        if wl.sum() >= min_walled and frac[i] >= min_frac:
            aw = np.maximum(a[wl], 0.5); ok[i] = True
            a_res[i] = float(np.median(aw)); a_comp[i] = float(np.sqrt(np.mean(aw ** 2)))
    return dict(resistance=a_res, compliance=a_comp, profiled=ok, walled_fraction=frac, segmented=rg)
