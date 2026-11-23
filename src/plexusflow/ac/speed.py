"""Clock measurements between tiles: the speed pulse of the same vessel in two recordings.

A vessel recorded in two tiles shows the same heartbeat in both movies.  The measurement is made on the velocity field itself, without
a profile fit and without a flow direction:

1. Geometry.  The vessel path (resampled every STEP px in the mosaic) is mapped into each tile with the stitching transforms; at each
   path point the optical-flow velocity (the pipeline's settings, frame spacing 1) is sampled at N_ACROSS positions across the vessel
   (± WIDTH_FRAC · radius), projected on the path tangent and averaged — one value per path point per frame (``along_vessel_velocity``).
2. Pooling.  For a pair of tiles, u(t) is the average over the path points that lie inside BOTH frames (at least MIN_POINTS), so both
   recordings describe the same piece of vessel.
3. Speed.  |u(t)| is fitted with its mean and K_FIT harmonics at the tile's heart rate.  The magnitude makes the measurement
   independent of the direction in which the recording reads the flow (some recordings — on 21S mostly poorly focused ones — read a
   vessel's flow in the opposite direction; their signed harmonics are then half a cycle off, their speed pulses are not).
4. Reversal.  A magnitude is only a faithful pulse if the velocity keeps its sign through the beat.  A recording whose pooled velocity
   (smoothed over REV_SMOOTH frames) spends at least REV_FRAC of the frames on each side of zero is marked reversing; pairs with a
   reversing recording are not used for the clocks.
5. Pair.  d = arg(ŝ_a,1 · conj ŝ_b,1) estimates ψ_a − ψ_b; variance τ_a² + τ_b², τ² = 2σ²/|ŝ_1|², σ² = residual variance / frames.

Driver: ``scripts/speed_pairs.py`` (the one step that reads the raw movies) → results/speed_pairs_v1.pkl; ``pulse.tile_clocks`` fits the
clocks.
"""
from __future__ import annotations
import inspect
import numpy as np

from ..reconcile.dc import K_FIT, FS_HZ

STEP = 2.0          #: px (mosaic) between path points
N_ACROSS = 7        #: samples across the vessel at each path point
WIDTH_FRAC = 0.8    #: … spanning ± this fraction of the vessel radius
MIN_POINTS = 5      #: common path points a tile pair needs (10 px of vessel)
REV_FRAC = 0.15     #: a recording is 'reversing' if its velocity is on each side of zero for at least this fraction of the frames
REV_SMOOTH = 5      #: frames of the moving average applied before that test
RESULT = "speed_pairs_v1"     #: results/<RESULT>.pkl written by scripts/speed_pairs.py


def flow_settings():
    """Farnebäck parameters and background percentile of the pipeline's per-vessel optical flow (``compute_of_profile_and_Q``)."""
    from ..analyze.profile import compute_of_profile_and_Q
    p = inspect.signature(compute_of_profile_and_Q).parameters
    fb = dict(pyr_scale=p["fb_pyr_scale"].default, levels=p["fb_levels"].default, winsize=p["fb_winsize"].default,
              iterations=p["fb_iterations"].default, poly_n=p["fb_poly_n"].default, poly_sigma=p["fb_poly_sigma"].default, flags=0)
    return fb, p["bg_percentile"].default


def path_points(path, step=STEP):
    """Points every `step` px along a vessel path ((n, ≥2) x, y in the mosaic), in the path's orientation; None if too short."""
    p = np.asarray(path, float)[:, :2]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if sl[-1] < 2 * step:
        return None
    s = np.arange(step / 2, sl[-1], step); return np.c_[np.interp(s, sl, p[:, 0]), np.interp(s, sl, p[:, 1])]


def tile_layout(points, radii, to_tile, radius_to_tile, shape, min_inside=0.2):
    """{edge: (rc (n, 2) tile [row, col] of the path points, radius in tile px)} for the vessels with at least `min_inside` of their
    path points inside the frame.  points {edge: (n, 2) mosaic x, y}; to_tile(xy) → [row, col]; radius_to_tile(r) → px."""
    hh, ww = shape; out = {}
    for e, p in points.items():
        rc = np.asarray(to_tile(p))
        if np.mean((rc[:, 0] >= 0) & (rc[:, 0] < hh) & (rc[:, 1] >= 0) & (rc[:, 1] < ww)) >= min_inside:
            out[e] = (rc, float(radius_to_tile(radii[e])))
    return out


def along_vessel_velocity(frames, layout):
    """Per vessel: (indices of the path points whose whole cross-section is inside the frame, U (T, n_valid) float16), U = velocity
    along the path tangent, averaged across the vessel, in px/frame.  frames: (T + 1, H, W) movie."""
    import cv2
    from ..analyze.profile import _normalise_for_of
    fb, bg = flow_settings(); V = _normalise_for_of(np.asarray(frames, np.float32), bg); T = len(V) - 1; hh, ww = V.shape[1:]
    ir, ic, tr, tc, valid = [], [], [], [], {}
    for e, (rc, R) in layout.items():
        tg = np.gradient(rc, axis=0); tg /= np.maximum(np.hypot(*tg.T), 1e-9)[:, None]; nr = np.c_[-tg[:, 1], tg[:, 0]]
        q = rc[:, None, :] + np.linspace(-WIDTH_FRAC * R, WIDTH_FRAC * R, N_ACROSS)[None, :, None] * nr[:, None, :]
        ok = np.all((q[..., 0] >= 2) & (q[..., 0] <= hh - 3) & (q[..., 1] >= 2) & (q[..., 1] <= ww - 3), axis=1)
        if ok.sum() < 3:
            continue
        valid[e] = np.flatnonzero(ok); q = q[ok]
        ir.append(np.round(q[..., 0]).astype(int)); ic.append(np.round(q[..., 1]).astype(int)); tr.append(tg[ok, 0]); tc.append(tg[ok, 1])
    if not valid:
        return {}
    IR, IC, TR, TC = np.concatenate(ir), np.concatenate(ic), np.concatenate(tr), np.concatenate(tc); U = np.empty((T, len(TR)), np.float32)
    for i in range(T):
        fl = cv2.calcOpticalFlowFarneback(V[i], V[i + 1], None, **fb)                         # [..., 0] = d col, [..., 1] = d row
        U[i] = (fl[IR, IC, 1] * TR[:, None] + fl[IR, IC, 0] * TC[:, None]).mean(1)
    cut = np.cumsum([0] + [len(v) for v in valid.values()])
    return {e: (valid[e], U[:, cut[j]:cut[j + 1]].astype(np.float16)) for j, e in enumerate(valid)}


def harmonic_fit(x, f0, fs=FS_HZ, K=K_FIT):
    """Least-squares mean + K harmonics of a series: (a0, H (K,) complex with x ≈ a0 + Σ Re[H_n e^{2πi n f0 t}], σ² = variance of a0)."""
    x = np.asarray(x, float); T = len(x); t = np.arange(T) / fs
    X = np.column_stack([np.ones(T)] + [g(2 * np.pi * k * f0 * t) for k in range(1, K + 1) for g in (np.cos, np.sin)])
    b, *_ = np.linalg.lstsq(X, x, rcond=None); r = x - X @ b
    return float(b[0]), np.array([b[2 * k - 1] - 1j * b[2 * k] for k in range(1, K + 1)]), float(r @ r) / (T - 2 * K - 1) / T


def reverses(u, frac=REV_FRAC, smooth=REV_SMOOTH):
    """Does the (smoothed) velocity spend at least `frac` of the frames on each side of zero?"""
    us = np.convolve(np.asarray(u, float), np.ones(smooth) / smooth, "valid"); return bool(min(np.mean(us > 0), np.mean(us < 0)) >= frac)


def speed_pairs(V, f0):
    """Clock measurements from every vessel seen in two tiles.  V {tile: along_vessel_velocity output}; f0 {tile: Hz}.
    Returns a dict of arrays, one entry per (vessel, tile pair): edge (n, 2), vessel (index), a, b (tiles), d (rad; estimates
    ψ_a − ψ_b), var (rad²), reversing (either recording reverses), n_points (common path points), and edges (list of vessels)."""
    byv = {}
    for t in sorted(V):
        for e in V[t]:
            byv.setdefault(e, []).append(t)
    rows, edges = [], []
    for e, ts in byv.items():
        if len(ts) < 2:
            continue
        iv = len(edges); edges.append(e)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]; va, Ua = V[a][e]; vb, Ub = V[b][e]; S = np.intersect1d(va, vb)
                if len(S) < MIN_POINTS:
                    continue
                ua = Ua[:, np.isin(va, S)].astype(np.float64).mean(1); ub = Ub[:, np.isin(vb, S)].astype(np.float64).mean(1)
                (_, Ha, sa), (_, Hb, sb) = harmonic_fit(np.abs(ua), f0[a]), harmonic_fit(np.abs(ub), f0[b])
                if abs(Ha[0]) == 0 or abs(Hb[0]) == 0:
                    continue
                rows.append((iv, a, b, float(np.angle(Ha[0] * np.conj(Hb[0]))), 2 * sa / abs(Ha[0]) ** 2 + 2 * sb / abs(Hb[0]) ** 2,
                             reverses(ua) or reverses(ub), len(S)))
    r = np.array(rows, float).reshape(-1, 7)
    return dict(edges=edges, vessel=r[:, 0].astype(int), a=r[:, 1].astype(int), b=r[:, 2].astype(int), d=r[:, 3], var=r[:, 4],
                reversing=r[:, 5].astype(bool), n_points=r[:, 6].astype(int))
