"""Centerline geometry + OF-driven refinement.

Given an initial (row, col) polyline through a vessel, compute arc
length + local tangent/normal, then optionally refine it against a
mean optical-flow field so it tracks the true velocity ridge.

Only the ``parabola_peak`` refinement mode is included — the older
argmax-based ``ridge_parallel`` and ``velocity_ridge`` variants from
PerTileFlow were dropped since they were only kept for A/B testing
against parabola_peak, which won.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree


# ── Geometry ────────────────────────────────────────────────────────

def centerline_geometry(pts: np.ndarray):
    """Compute arc lengths + unit tangents + CW normals.

    Parameters
    ----------
    pts : (N, 2) float [row, col]

    Returns
    -------
    s        : (N,) cumulative arc length
    tangents : (N, 2) unit tangents [row_component, col_component]
    normals  : (N, 2) CW normals = (t_col, -t_row)
    """
    pts = np.asarray(pts, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    raw = np.empty_like(pts)
    raw[1:-1] = pts[2:] - pts[:-2]
    raw[0]    = pts[1]  - pts[0]
    raw[-1]   = pts[-1] - pts[-2]
    nrm = np.sqrt((raw ** 2).sum(axis=1, keepdims=True))
    tangents = raw / np.maximum(nrm, 1e-8)
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    return s, tangents, normals


# Alias so intra-package callers can keep the underscore-prefixed name.
_centerline_geometry = centerline_geometry


# ── Parabola-peak CL refinement ─────────────────────────────────────

def _coarse_rigid_shift(
    flow_avg, cl_prior, R_prior, H, W, mask=None,
    n_bins: int = 21, fit_fraction: float = 0.85,
    max_shift_frac: float = 0.7,
):
    """Coarse pre-pass for CL refinement.

    Aggregate the axial velocity profile over ALL slices of the CL,
    fit ONE parabola to the aggregated profile, and shift every CL
    point by the resulting peak location along its local normal.

    Rationale: per-slice parabola fits are noisy on real vessels, and
    on curved CLs the local normals point in different directions —
    so per-slice shifts can have opposite signs and cancel to ~0 when
    smoothed, even when the vessel is clearly off-axis.  Fitting one
    parabola to the AGGREGATED profile has much better SNR and gives
    a stable global-shift estimate, which is what your eye does when
    looking at a mean-velocity heatmap.

    Returns
    -------
    (cl_shifted, coarse_shift_px)
        The rigid-shifted CL and the applied shift.  Returns the
        input CL unchanged (and 0.0) if the fit is degenerate.
    """
    _, tang, norm = _centerline_geometry(cl_prior)
    N = len(cl_prior)
    bin_edges = np.linspace(-R_prior, R_prior, n_bins + 1)
    rho = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prof_sum = np.zeros(n_bins)
    prof_cnt = np.zeros(n_bins, dtype=int)
    for i in range(N):
        row_c, col_c = cl_prior[i]
        n_r, n_c = norm[i]
        t_r, t_c = tang[i]
        for j, rho_j in enumerate(rho):
            rr = row_c + rho_j * n_r
            cc = col_c + rho_j * n_c
            if not (0 <= rr < H - 1 and 0 <= cc < W - 1):
                continue
            r0, c0 = int(rr), int(cc)
            fr, fc = rr - r0, cc - c0
            vx = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 0]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 0]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 0]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 0])
            vy = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 1]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 1]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 1]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 1])
            if mask is not None:
                mr, mc = int(round(rr)), int(round(cc))
                if not (0 <= mr < mask.shape[0]
                        and 0 <= mc < mask.shape[1] and mask[mr, mc]):
                    continue
            v_axial = vy * t_r + vx * t_c
            prof_sum[j] += v_axial
            prof_cnt[j] += 1
    profile = np.where(prof_cnt > 0, prof_sum / np.maximum(prof_cnt, 1), np.nan)
    fit_ok = np.abs(rho) <= R_prior * fit_fraction
    r_fit = rho[fit_ok]
    v_fit = profile[fit_ok]
    good = np.isfinite(v_fit)
    if good.sum() < 5:
        return cl_prior, 0.0
    X = np.column_stack([np.ones(good.sum()),
                          r_fit[good], r_fit[good] ** 2])
    try:
        coeffs, *_ = np.linalg.lstsq(X, v_fit[good], rcond=None)
    except Exception:
        return cl_prior, 0.0
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    if abs(c) < 1e-9:
        return cl_prior, 0.0
    rho_peak = -b / (2.0 * c)
    v_peak = a - b * b / (4.0 * c)
    if v_peak * c >= 0:            # concavity/peak sign mismatch
        return cl_prior, 0.0
    max_shift = max_shift_frac * R_prior
    rho_peak = float(np.clip(rho_peak, -max_shift, max_shift))
    if abs(rho_peak) < 0.5:        # too small to matter
        return cl_prior, float(rho_peak)
    cl_new = cl_prior.copy().astype(float)
    cl_new[:, 0] += rho_peak * norm[:, 0]
    cl_new[:, 1] += rho_peak * norm[:, 1]
    return cl_new, float(rho_peak)


def _profile_offcenter(flow_avg, cl, R_prior, H, W, mask=None,
                       n_bins=21, fit_fraction=0.85):
    """|velocity|-weighted centroid offset (px) of the aggregated cross-section
    profile: 0 = the CL is centered on the flow, large |value| = the CL sits off
    the true axis.  Unlike a parabola-vertex estimate this is robust to
    non-parabolic (monotonic / bimodal) profiles — which is exactly the case a
    mis-refined CL produces — so it can tell whether refinement improved or
    worsened the centering."""
    _, tang, norm = _centerline_geometry(cl)
    N = len(cl)
    bin_edges = np.linspace(-R_prior, R_prior, n_bins + 1)
    rho = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prof_sum = np.zeros(n_bins)
    prof_cnt = np.zeros(n_bins, dtype=int)
    for i in range(N):
        row_c, col_c = cl[i]
        n_r, n_c = norm[i]
        t_r, t_c = tang[i]
        for j, rho_j in enumerate(rho):
            rr = row_c + rho_j * n_r
            cc = col_c + rho_j * n_c
            if not (0 <= rr < H - 1 and 0 <= cc < W - 1):
                continue
            r0, c0 = int(rr), int(cc)
            fr, fc = rr - r0, cc - c0
            vx = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 0]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 0]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 0]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 0])
            vy = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 1]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 1]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 1]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 1])
            if mask is not None:
                mr, mc = int(round(rr)), int(round(cc))
                if not (0 <= mr < mask.shape[0]
                        and 0 <= mc < mask.shape[1] and mask[mr, mc]):
                    continue
            prof_sum[j] += (vy * t_r + vx * t_c)
            prof_cnt[j] += 1
    prof = np.where(prof_cnt > 0, prof_sum / np.maximum(prof_cnt, 1), np.nan)
    fit = np.abs(rho) <= R_prior * fit_fraction
    w = np.abs(prof)
    ok = fit & np.isfinite(w)
    if ok.sum() < 3 or float(w[ok].sum()) < 1e-9:
        return 0.0
    return float(np.sum(rho[ok] * w[ok]) / np.sum(w[ok]))


def ridge_track_centerline(flow_avg, cl_prior, R_prior, H, W, mask=None,
                           sample_frac=1.0, offset_smooth=6.0,
                           max_shift_frac=1.0, iterations=3,
                           spatial_smooth_frac: float = 0.25,
                           fix_tangent_to_prior: bool = True):
    """OF-driven centerline refinement by LOCAL ridge tracking.

    Replaces the old coarse-rigid-shift + per-slice-snap.  At each CL point,
    sample the axial |v| along the perpendicular over ±sample_frac·R, weight by
    a proximity window (Gaussian, width R) so the estimate prefers the local
    flow ridge and can't jump to a neighbouring vessel, and move the point to
    the windowed centroid of that weighted profile.  Offsets are interpolated
    over ambiguous slices, smoothed along the arc for coherence, and capped to
    ±max_shift_frac·R.  Iterated a few times.

    This tracks the actual velocity ridge locally (handling curvature and
    non-rigid warps — a whole-network graph CL rarely maps rigidly onto one
    tile's video), instead of applying one global rigid shift that a curved or
    neighbour-adjacent vessel fools.

    Returns ``(cl_refined, offsets_smooth)``.
    """
    cl = np.asarray(cl_prior, dtype=float).copy()
    offs_final = np.zeros(len(cl))
    # Spatial-smooth the flow field before ridge tracking.  On low-SNR
    # (slow / dim) vessels the per-pixel OF is noise-dominated and the
    # centroid per slice wanders; a σ ≈ R/4 Gaussian on (u, v) averages
    # away the noise while preserving the vessel-scale ridge structure.
    # Set spatial_smooth_frac=0 to disable.
    if spatial_smooth_frac and spatial_smooth_frac > 0:
        _sig = max(1.0, spatial_smooth_frac * float(R_prior))
        _u_s = gaussian_filter(flow_avg[..., 0], _sig, mode="nearest")
        _v_s = gaussian_filter(flow_avg[..., 1], _sig, mode="nearest")
        flow_avg = np.stack([_u_s, _v_s], axis=-1).astype(np.float32, copy=False)
    # Vessel-level peakiness — the ridge tracker only helps when there IS a
    # ridge.  On a blunt / plug / flat-topped vessel the perpendicular
    # cross-sections are plateaux, the per-slice centroid drifts (toward
    # brighter tissue on one side or a companion vessel), and the smoothed
    # + iterated result is a systematically-biased CL that reads worse than
    # the input.  Compute the median peak/mean ratio across all valid
    # slices ON THE FIRST ITERATION; if it's below ``vessel_peakiness_veto``
    # the whole refinement is unreliable — revert to the prior CL, same
    # spirit as the >270° net-turn guard below.
    vessel_peakiness_veto: float = 1.2
    peakiness_scores: list[float] = []
    # Anchor the projection tangent to the input CL (graph anatomy) rather
    # than recompute it from the drifting `cl` each iteration.  And use ONE
    # global tangent (the median direction across all CL points) instead of
    # per-point tangents — a short vessel segment between junctions has a
    # single anatomical axis; per-point tangents wiggle from CL noise and
    # give unstable projections that then bias the ridge search differently
    # at each slice.  A single global axis makes the projected magnitude a
    # well-defined "how much OF flows along the vessel" everywhere.
    _, _tang_pp, _ = _centerline_geometry(np.asarray(cl_prior, dtype=float))
    # Median tangent (circular-mean equivalent): take unit vectors, sum, renormalize.
    # Guard against opposite-pointing tangents by first sign-aligning to the
    # first tangent (a robust choice on curly CLs).
    _t0 = _tang_pp[0]
    _signs = np.sign(_tang_pp @ _t0)
    _signs[_signs == 0] = 1.0
    _tsum = (_tang_pp * _signs[:, None]).sum(axis=0)
    _tnorm = float(np.hypot(*_tsum))
    _tang_global = (_tsum / _tnorm) if _tnorm > 1e-9 else _t0
    _tang_anchor = np.tile(_tang_global, (len(_tang_pp), 1))
    for _ in range(max(1, int(iterations))):
        _, tang_curr, norm = _centerline_geometry(cl)
        tang = _tang_anchor if fix_tangent_to_prior else tang_curr
        N = len(cl)
        sR = sample_frac * R_prior
        n_bins = int(2 * sR) | 1                 # ~1px bins, odd
        rho = np.linspace(-sR, sR, n_bins)
        win = np.exp(-(rho / max(R_prior, 1e-6)) ** 2 / 2.0)
        offs = np.zeros(N)
        valid = np.zeros(N, dtype=bool)
        for i in range(N):
            row_c, col_c = cl[i]
            n_r, n_c = norm[i]
            t_r, t_c = tang[i]
            a = np.full(n_bins, np.nan)
            for j, rho_j in enumerate(rho):
                rr = row_c + rho_j * n_r
                cc = col_c + rho_j * n_c
                if not (0 <= rr < H - 1 and 0 <= cc < W - 1):
                    continue
                r0, c0 = int(rr), int(cc)
                fr, fc = rr - r0, cc - c0
                vx = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 0]
                      + fr * (1 - fc) * flow_avg[r0 + 1, c0, 0]
                      + (1 - fr) * fc * flow_avg[r0, c0 + 1, 0]
                      + fr * fc * flow_avg[r0 + 1, c0 + 1, 0])
                vy = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 1]
                      + fr * (1 - fc) * flow_avg[r0 + 1, c0, 1]
                      + (1 - fr) * fc * flow_avg[r0, c0 + 1, 1]
                      + fr * fc * flow_avg[r0 + 1, c0 + 1, 1])
                a[j] = abs(vy * t_r + vx * t_c)
                # Mask gate — leave a[j] = NaN for bins outside the vessel
                # mask so the centroid can't be pulled by a companion vessel
                # or spillover flow within the ±sample_frac·R window.  Same
                # gate profile_flux_scalar already applies for its profile
                # sampling; ridge-track was silently omitting it.
                if mask is not None:
                    mr, mc = int(round(rr)), int(round(cc))
                    if not (0 <= mr < mask.shape[0]
                            and 0 <= mc < mask.shape[1]
                            and mask[mr, mc]):
                        a[j] = np.nan
            ok = np.isfinite(a)
            if ok.sum() < 5:
                continue
            # Peakiness gate — the amplitude-weighted centroid only
            # localizes when the perpendicular has a distinct peak.  On a
            # plug / blunt / plateau profile the "ridge" is a 2-D region
            # and the centroid drifts (toward brighter tissue on one side,
            # a companion vessel, or asymmetric contamination).  Require
            # max(|v|) / mean(|v|) > 1.5 on the valid bins — a Poiseuille
            # parabola has this ratio ≈ 1.5–2 and a plug has ≈ 1.  Below
            # that, leave the CL point where it was.
            _av = a[ok]
            _peakiness = (float(_av.max()) / float(_av.mean())
                          if float(_av.mean()) > 1e-9 else 0.0)
            if len(peakiness_scores) < N:   # collect first-iteration only
                peakiness_scores.append(_peakiness)
            if _peakiness > 0.0 and _peakiness < 1.2:
                continue
            # PROP-A: pick the ARGMAX of |v_axial| along the perpendicular,
            # weighted by a proximity window centred on the graph-CL position.
            # Weighted-centroid (previous behaviour) averaged multiple peaks
            # into a mid-position and consistently under-corrected when the
            # true ridge sat off-axis by more than a couple pixels — visible
            # on tile 25 edge (70294, 70422) where the vessel bulge is ~5 px
            # south of the graph CL but the centroid barely moved.  Argmax
            # picks the strongest peak in the ±R search window.
            aw = np.where(ok, a, 0.0) * win
            if float(aw.max()) < 1e-9:
                continue
            offs[i] = float(rho[int(np.argmax(aw))])
            valid[i] = True
        if not valid.any():
            break
        if not valid.all():
            idx = np.arange(N)
            offs = np.interp(idx, idx[valid], offs[valid])
        # Cap the smoothing so short vessels aren't over-smoothed.
        _os = min(offset_smooth, max(N / 4.0, 1.0))
        if _os > 0 and N > 3:
            offs = gaussian_filter1d(offs, _os)
        offs = np.clip(offs, -max_shift_frac * R_prior, max_shift_frac * R_prior)
        # Apply the PERPENDICULAR offset only — this moves each point onto the
        # ridge without changing arc length.  Do NOT gaussian-smooth the CL xy
        # coordinates afterwards: on a short line that pulls the endpoints
        # inward and COLLAPSES it toward the middle (the offset smoothing above
        # already provides a coherent, length-preserving correction).
        cl = cl + offs[:, None] * norm
        offs_final = offs
    # Vessel-level peakiness veto — see `vessel_peakiness_veto` above.  If the
    # median across-slice peakiness (max|v|/mean|v| on the perpendicular) is
    # below the threshold, the vessel is fundamentally blunt / plug-like — no
    # localizable ridge to track — and the tracker's smoothed offsets are just
    # systematically biased noise.  Revert to the input CL entirely (same as
    # the >270° net-turn guard below).
    if peakiness_scores:
        _med_peak = float(np.median(peakiness_scores))
        if _med_peak < vessel_peakiness_veto:
            cl = np.asarray(cl_prior, dtype=float).copy()
            offs_final = np.zeros(len(cl))
    # Curvature guard.  On a large-R blob or junction the "flow ridge" is a 2D
    # region, not a line, and the per-slice centroid tracking can WIND the CL
    # around it into an implausible hook — observed: a 70° graph CL bent to 409°
    # (>a full turn), whose radius of curvature then drops below the cross-section
    # sampling half-width, so slices sweep a circle and the radial profile inverts
    # (negative taper).  A real vessel segment between junctions turns <~180°.  If
    # the refinement adds a lot of turning AND the result is implausibly bent, the
    # offsets are untrustworthy → revert to the prior CL (still de-kinked below).
    def _net_turn_deg(a):
        # NET tangent rotation of a lightly-smoothed CL — robust to per-point
        # jaggedness (a jagged-but-straight CL has near-zero net turning even
        # though the sum of |angle changes| is large), so it flags only genuine
        # winding (a hook/loop rotates monotonically → large net turn).
        a = np.asarray(a, dtype=float)
        if len(a) < 5:
            return 0.0
        _sig = max(1.0, len(a) / 20.0)
        xs = gaussian_filter1d(a[:, 0], _sig, mode='nearest')
        ys = gaussian_filter1d(a[:, 1], _sig, mode='nearest')
        d = np.diff(np.column_stack([xs, ys]), axis=0)
        ang = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
        return float(abs(np.degrees(ang[-1] - ang[0])))
    # A real vessel segment between junctions rarely turns >180° net; >270°
    # (three-quarter turn) means the tracker wound around a blob → untrustworthy.
    if _net_turn_deg(cl) > 270.0:
        cl = np.asarray(cl_prior, dtype=float).copy()
        offs_final = np.zeros(len(cl))
    # Final path smoothing + uniform resampling.  Per-slice argmax leaves
    # small kinks (few-pixel oscillations) and occasional local loops that
    # per-slice offset-smoothing can't fix once applied.  Smooth the xy
    # coords along arc with σ scaled to R (~R/4, capped to ~arc-length/4 so
    # short vessels aren't collapsed), then re-sample to uniform 1px arc
    # spacing.  Uses mode='nearest' so endpoints don't drift inward.
    if len(cl) > 4:
        # arc-based smoothing sigma: R/4 pixels (or shorter if vessel is short)
        _arc = float(np.sum(np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1]))))
        _xs = float(min(max(0.25 * R_prior, 1.5), max(_arc / 4.0, 1.5)))
        cl = np.column_stack([
            gaussian_filter1d(cl[:, 0], _xs, mode='nearest'),
            gaussian_filter1d(cl[:, 1], _xs, mode='nearest')])
        # Re-uniform arc spacing (~1 px) so downstream slice bookkeeping is even
        _d = np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1]))
        _s = np.concatenate([[0.0], np.cumsum(_d)])
        _tot = float(_s[-1])
        if _tot >= 2:
            _n = int(np.clip(round(_tot) + 1, max(5, len(cl)), 150))
            _sn = np.linspace(0.0, _tot, _n)
            cl = np.column_stack([np.interp(_sn, _s, cl[:, 0]),
                                    np.interp(_sn, _s, cl[:, 1])])
    return cl, offs_final


def _snap_to_parabola_peak_pass(
    flow_avg, cl_prior, R_prior, H, W, dt=1, smooth_sigma=5.0,
    max_shift_frac=0.5, mask=None,
    n_bins=17, fit_fraction=0.85,
    cl_post_smooth_sigma=0.0,
):
    """Single-pass parabola-peak refinement.

    At each CL point sample the OF perpendicular slice, fit a quadratic
    ``v(ρ) = a + b·ρ + c·ρ²`` via linear least-squares, and extract the
    peak location ``ρ_peak = -b / (2c)``.  Smooth ``ρ_peak(arc)`` and
    apply as the per-point CL shift.

    Advantages over argmax:
      * subpixel by construction (peak from the fit, not from a discrete grid)
      * uses the entire slice (~15+ pixels)
      * naturally rejects non-Poiseuille slices via the fit-residual gate
    """
    _, tang, norm = _centerline_geometry(cl_prior)
    N = len(cl_prior)
    max_offset = max_shift_frac * R_prior

    bin_edges = np.linspace(-R_prior, R_prior, n_bins + 1)
    rho = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fit_ok = np.abs(rho) <= R_prior * fit_fraction
    rho_fit = rho[fit_ok]
    X = np.column_stack([np.ones_like(rho_fit), rho_fit, rho_fit ** 2])

    offsets_raw = np.zeros(N)
    offset_valid = np.zeros(N, dtype=bool)

    for i in range(N):
        row_c, col_c = cl_prior[i]
        n_r, n_c = norm[i]
        t_r, t_c = tang[i]
        v_slice = np.full(n_bins, np.nan)
        for j, rho_j in enumerate(rho):
            rr = row_c + rho_j * n_r
            cc = col_c + rho_j * n_c
            if not (0 <= rr < H - 1 and 0 <= cc < W - 1):
                continue
            r0, c0 = int(rr), int(cc)
            fr, fc = rr - r0, cc - c0
            vx = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 0]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 0]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 0]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 0])
            vy = ((1 - fr) * (1 - fc) * flow_avg[r0, c0, 1]
                  + fr * (1 - fc) * flow_avg[r0 + 1, c0, 1]
                  + (1 - fr) * fc * flow_avg[r0, c0 + 1, 1]
                  + fr * fc * flow_avg[r0 + 1, c0 + 1, 1])
            if mask is not None:
                mr, mc = int(round(rr)), int(round(cc))
                if not (0 <= mr < mask.shape[0]
                        and 0 <= mc < mask.shape[1] and mask[mr, mc]):
                    continue
            v_slice[j] = (vy * t_r + vx * t_c) / max(dt, 1)

        v_fit = v_slice[fit_ok]
        good = np.isfinite(v_fit)
        if good.sum() < 5:
            continue
        try:
            coeffs, *_ = np.linalg.lstsq(X[good], v_fit[good], rcond=None)
        except Exception:
            continue
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        if abs(c) < 1e-9:
            continue
        rho_peak = -b / (2.0 * c)
        v_peak = a - b * b / (4.0 * c)
        if v_peak * c >= 0:                     # concavity/peak sign mismatch
            continue
        R_implied = float(np.sqrt(max(-v_peak / c, 1e-9)))
        if not (0.3 * R_prior < R_implied < 3.0 * R_prior):
            continue
        pred = coeffs[0] + coeffs[1] * rho_fit[good] + coeffs[2] * rho_fit[good] ** 2
        rss = float(np.sum((v_fit[good] - pred) ** 2))
        var = float(np.var(v_fit[good]) * len(v_fit[good]))
        r2 = 1.0 - rss / max(var, 1e-9)
        if r2 < 0.3:
            continue
        offsets_raw[i] = float(np.clip(rho_peak, -max_offset, max_offset))
        offset_valid[i] = True

    # Fill invalid slices by nearest-valid interpolation before smoothing
    if offset_valid.any() and not offset_valid.all():
        idx = np.arange(N)
        offsets_raw = np.interp(idx, idx[offset_valid],
                                  offsets_raw[offset_valid])
    elif not offset_valid.any():
        return cl_prior.copy(), offsets_raw, offsets_raw.copy()

    if smooth_sigma > 0 and N > 3:
        offsets_smooth = gaussian_filter1d(offsets_raw, smooth_sigma)
    else:
        offsets_smooth = offsets_raw.copy()
    offsets_smooth = np.clip(offsets_smooth, -max_offset, max_offset)

    # Safety: if mean shift is huge, refinement is unreliable — reset
    if float(np.mean(np.abs(offsets_smooth))) > R_prior / 3.0:
        offsets_smooth[:] = 0.0

    cl_refined = cl_prior + offsets_smooth[:, None] * norm
    if cl_post_smooth_sigma > 0 and N > 3:
        cl_refined = np.column_stack([
            gaussian_filter1d(cl_refined[:, 0], cl_post_smooth_sigma),
            gaussian_filter1d(cl_refined[:, 1], cl_post_smooth_sigma),
        ])
    return cl_refined, offsets_raw, offsets_smooth


def snap_to_parabola_peak(
    flow_avg, cl_prior, R_prior, H, W, dt=1, smooth_sigma=5.0,
    max_shift_frac=0.5, mask=None, n_bins=17, fit_fraction=0.85,
    iterations=1, shift_tol_px=1.0, cl_post_smooth_sigma=0.0,
):
    """Iterative wrapper for :func:`_snap_to_parabola_peak_pass`.

    Runs up to ``iterations`` passes, breaking when the max per-point
    shift falls below ``shift_tol_px``.
    """
    cl = np.asarray(cl_prior, dtype=float).copy()
    offsets_raw_last = np.zeros(len(cl))
    offsets_smooth_last = np.zeros(len(cl))
    for _ in range(max(1, int(iterations))):
        cl_new, offsets_raw_last, offsets_smooth_last = (
            _snap_to_parabola_peak_pass(
                flow_avg, cl, R_prior, H, W, dt=dt,
                smooth_sigma=smooth_sigma, max_shift_frac=max_shift_frac,
                mask=mask, n_bins=n_bins, fit_fraction=fit_fraction,
                cl_post_smooth_sigma=cl_post_smooth_sigma))
        max_shift = float(np.max(np.abs(offsets_smooth_last)))
        cl = cl_new
        if max_shift < shift_tol_px:
            break
    return cl, offsets_raw_last, offsets_smooth_last


_snap_to_parabola_peak = snap_to_parabola_peak         # legacy alias


# ── Top-level refinement (compute OF once, refine CL, compute tangents) ──

def refine_centerline(
    frames_u8, cl_prior, R_prior, fb_kwargs, dt=1,
    n_warmup=80,
    f0_hz=None, frame_dt_s=None,
    tangent_smooth_sigma=5.0,
    use_flow_tangent=True,
    ridge_mask=None,
    ridge_iterations=1,
    ridge_shift_tol_px=1.0,
    ridge_offset_smooth_sigma=4.0,
    mask_frac=2.0,
):
    """Refine a prior centerline using OF-derived symmetry.

    Steps:
      1. Compute the mean optical-flow field (systolic frames preferred
         when ``f0_hz`` and ``frame_dt_s`` are provided).
      2. Iterate ``parabola_peak`` refinement to snap the CL onto the
         local velocity ridge.
      3. Aggregate a flow-derived tangent field and smooth it along arc
         (via ``tangent_smooth_sigma``) so downstream cross-section
         binning uses stable perpendiculars.

    Returns
    -------
    dict with:
      * ``cl_refined``  — (N, 2) refined centerline
      * ``rr_in`` / ``cc_in`` / ``rho_in`` — vessel-pixel indices + signed ρ
      * ``t_row_in`` / ``t_col_in`` — per-pixel tangent components
      * ``vfield_mean`` — (H, W, 2) mean OF field (px/frame)
      * ``shift_applied`` — mean of the smoothed per-point offsets
      * ``n_vessel_px`` — count of vessel pixels captured
    """
    import cv2

    T, H, W = frames_u8.shape
    n_warmup = min(n_warmup, T - dt)

    # 1. Mean flow field — prefer systolic-phase frames when we know f0
    if f0_hz is not None and frame_dt_s is not None and f0_hz > 0:
        phi = (np.arange(T - dt) * frame_dt_s * f0_hz) % 1.0
        systolic_idx = np.where(phi < 0.3)[0]
        if len(systolic_idx) >= n_warmup // 2:
            indices = systolic_idx[np.linspace(
                0, len(systolic_idx) - 1, n_warmup, dtype=int)]
        else:
            indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)
    else:
        indices = np.linspace(0, T - dt - 1, n_warmup, dtype=int)

    flow_sum = np.zeros((H, W, 2), dtype=np.float64)
    for idx in indices:
        flow = cv2.calcOpticalFlowFarneback(
            frames_u8[idx], frames_u8[idx + dt], None, **fb_kwargs)
        flow_sum += flow
    flow_avg = flow_sum / len(indices)

    # 2. OF-driven refinement by LOCAL RIDGE TRACKING.  Replaces the old
    #    coarse-rigid-shift + per-slice-snap: those applied one GLOBAL rigid
    #    perpendicular shift (fit a single parabola to the aggregated profile,
    #    translate the whole CL), which a curved or neighbour-adjacent vessel
    #    fools — the aggregate is skewed and the whole line slides off toward a
    #    neighbouring vessel.  The ridge tracker instead moves each CL point to
    #    the local velocity-ridge centroid (proximity-windowed so it can't jump
    #    to a neighbour), arc-smoothed for coherence and capped to ±R — handling
    #    curvature and non-rigid warps (a whole-network graph CL rarely maps
    #    rigidly onto one tile's video).  Validated on real vessels: recenters
    #    an off-axis graph CL where the rigid method drifted.
    cl_refined, offsets_smooth = ridge_track_centerline(
        flow_avg, cl_prior, R_prior, H, W, mask=ridge_mask,
        offset_smooth=max(ridge_offset_smooth_sigma, 6.0),
        max_shift_frac=1.0, iterations=max(2, int(ridge_iterations) + 1))
    # Diagnostics-compat fields (no separate coarse stage / revert any more).
    cl_coarse = np.asarray(cl_prior, dtype=float)
    coarse_shift = 0.0
    offsets_raw = offsets_smooth
    best_shift = float(np.mean(offsets_smooth)) if len(offsets_smooth) else 0.0
    reverted = False
    peak_prior = peak_ref = float("nan")

    # 3. Build vessel-pixel index + local tangents
    _, tang_r, _ = _centerline_geometry(cl_refined)
    cl_tree_r = cKDTree(cl_refined)
    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                              cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree_r.query(pts_g)
    mask_r = ((dist_g.reshape(H, W) <= R_prior * max(mask_frac, 1.2))
              & (nearest_g.reshape(H, W) > 0)
              & (nearest_g.reshape(H, W) < len(cl_refined) - 1))
    rr_in, cc_in = np.where(mask_r)
    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree_r.query(pts_in)

    if use_flow_tangent:
        flow_vr = flow_avg[rr_in, cc_in, 1]
        flow_vc = flow_avg[rr_in, cc_in, 0]
        flow_tang = np.column_stack([flow_vr, flow_vc])
        flow_mag = np.sqrt(flow_tang[:, 0] ** 2 + flow_tang[:, 1] ** 2)
        flow_mag_safe = np.maximum(flow_mag, 1e-8)
        flow_tang /= flow_mag_safe[:, None]
        # Orient consistently with geometric tangent
        cl_tang = tang_r[nearest]
        dot = (flow_tang * cl_tang).sum(axis=1)
        flow_tang[dot < 0] *= -1
        weak = flow_mag < 0.03
        flow_tang[weak] = cl_tang[weak]
        # Aggregate per CL point, smooth along arc, map back to pixels
        n_cl = len(cl_refined)
        tang_cl_sum = np.zeros((n_cl, 2))
        tang_cl_w = np.zeros(n_cl)
        for k in range(len(nearest)):
            j = nearest[k]
            w = float(flow_mag[k])
            tang_cl_sum[j] += w * flow_tang[k]
            tang_cl_w[j] += w
        has_data = tang_cl_w > 1e-8
        tang_cl_avg = np.copy(tang_r)
        tang_cl_avg[has_data] = (tang_cl_sum[has_data]
                                  / tang_cl_w[has_data, None])
        if tangent_smooth_sigma > 0 and n_cl > 3:
            tang_cl_avg[:, 0] = gaussian_filter1d(
                tang_cl_avg[:, 0], tangent_smooth_sigma)
            tang_cl_avg[:, 1] = gaussian_filter1d(
                tang_cl_avg[:, 1], tangent_smooth_sigma)
            nrm = np.sqrt((tang_cl_avg ** 2).sum(axis=1, keepdims=True))
            tang_cl_avg /= np.maximum(nrm, 1e-8)
        flow_tang = tang_cl_avg[nearest]
        t_row_in = flow_tang[:, 0]
        t_col_in = flow_tang[:, 1]
    else:
        t_row_in = tang_r[nearest, 0]
        t_col_in = tang_r[nearest, 1]

    n_row_in = t_col_in.copy()      # normal = (t_col, -t_row)
    n_col_in = -t_row_in.copy()
    rho_in = ((rr_in - cl_refined[nearest, 0]) * n_row_in
              + (cc_in - cl_refined[nearest, 1]) * n_col_in)

    vfield_mean = np.zeros((H, W, 2), dtype=np.float64)
    vfield_mean[..., 0] = flow_avg[..., 0] / max(dt, 1)
    vfield_mean[..., 1] = flow_avg[..., 1] / max(dt, 1)

    return dict(
        cl_refined=cl_refined,
        shift_applied=best_shift,
        rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
        t_row_in=t_row_in, t_col_in=t_col_in,
        n_vessel_px=int(len(rr_in)),
        vfield_mean=vfield_mean,
        # ── Refinement intermediates (for the pipeline-walkthrough figure).
        # Additive keys; callers that don't ask for them are unaffected.
        cl_initial=np.asarray(cl_prior, dtype=float).copy(),  # pre-refinement
        cl_coarse=np.asarray(cl_coarse, dtype=float),          # after rigid pre-pass
        coarse_shift=float(coarse_shift),                      # rigid shift (px)
        offsets_raw=offsets_raw,                               # per-slice peak offset (raw)
        offsets_smooth=offsets_smooth,                         # …after arc-smoothing
        flow_avg=flow_avg,                                     # warmup mean OF (px/frame·dt)
        mask=mask_r,                                           # refined vessel mask (H,W bool)
        reverted=reverted,                                     # refinement undone (made profile worse)
        refine_peak_prior=float(peak_prior),                   # graph-CL residual offset (px)
        refine_peak_ref=float(peak_ref),                       # refined-CL residual offset (px)
    )


_refine_centerline = refine_centerline    # legacy alias
