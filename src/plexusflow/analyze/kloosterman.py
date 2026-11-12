"""Kloosterman/Poelma per-cross-section scalar Q.

Fits a Poiseuille parabola per slice along the vessel centerline,
drops slices whose flow direction disagrees with the median (optionally a median-band or IQR rule), then
averages the survivors into a characteristic (Q_char, V_CL_char,
R_char).  Uses the joint (v0, R) free fit per slice — with a default
override to compute Q from ``R_prior`` (mask R) rather than the
per-slice fit R, which cuts down noise-driven bloat.

Assumptions:
  * Q is constant along the vessel (low Re, developed flow — yolk-sac
    Wo ~ 0.2 satisfies this).
  * Cross-section velocity profile is Poiseuille.
  * Outlier slices (branch-adjacent, mask-artifact, near-junction bulge)
    drop out via the robust filter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .centerline import _centerline_geometry


# The settings the batch uses (analyze/profile.py, mean-flow field already per frame, 60 ρ bins over ±2R) — for code that
# re-runs the estimator on the cached batch inputs (figures, scripts/kl_reaggregate.py).
PRODUCTION_KW = dict(dt=1, slice_window=2, n_bins=60, fit_fraction=1.0, outlier_mode="sign", R_slice_lo=0.5, R_slice_hi=2.0,
                     min_valid_slices=2, use_r_prior_area=False, v_incl_frac=0.25, min_r2=0.5, sample_frac=2.0)


def parabolicity(per_slice_r2):
    """Mean parabolic fit quality of a recording, p = mean over ALL cross-sections of clip(R², 0, 1); a section that never
    reached a fit (too few points, radius at its bound, peak below the profile mean) counts 0.  Continuous counterpart of
    the kept fraction f (rank correlation 0.95–0.97); it is the profile term of the reliability model since 2026-11-10."""
    r2 = np.asarray(per_slice_r2, float)
    return float(np.mean(np.where(np.isfinite(r2), np.clip(r2, 0.0, 1.0), 0.0))) if r2.size else 0.0


def kloosterman_scalar(
    flow_avg,
    cl,
    R_prior,
    dt: float = 1,
    slice_window: int = 2,
    n_bins: int = 17,
    fit_fraction: float = 0.85,
    outlier_mode: str = "sign",              # "sign" | "median_band" | "iqr"
    iqr_k: float = 1.5,
    median_lo: float = 0.3,
    median_hi: float = 2.5,
    min_valid_slices: int = 3,
    R_slice_lo: float = 0.3,
    R_slice_hi: float = 2.5,
    mask=None,
    fit_bound: str = "prior",                 # "prior" | "mask"
    edt=None,
    robust_loss: Optional[str] = None,
    use_r_prior_area: bool = False,
    # Kloosterman-thesis per-fit refinements:
    #  * v_incl_frac: include a cross-section point only if |v| exceeds this
    #    fraction of the centerline velocity — adaptive edge/mask-artifact
    #    rejection (thesis uses 0.25).  0 disables (pure geometric window).
    #  * require_v0_gt_mean: reject a fit whose vertex velocity does not
    #    exceed the mean of the fitted velocities (a degenerate/flat fit).
    v_incl_frac: float = 0.25,
    require_v0_gt_mean: bool = True,
    #  * sample_frac: sample each perpendicular out to sample_frac·R_prior (not
    #    just R_prior), so the per-slice velocity-extent detector can see where
    #    the flow actually ends when the mask R under-segments or the CL is
    #    offset.  The fit REGION (fit_fraction) and gates still key off R_prior.
    sample_frac: float = 2.0,
    #  * min_r2: reject a slice whose measured cross-section does not actually
    #    fit a parabola (coefficient of determination below this) — the direct
    #    "is this cross-section parabolic?" test that catches END / JUNCTION
    #    effects (merges, adjacent vessels, flow turning) by SHAPE rather than by
    #    arc position.  0 disables.
    min_r2: float = 0.5,
    #  * fit_center_offset: also fit a transverse offset ρ0 of the parabola
    #    (v0·(1 − ((ρ−ρ0)/R)²)), bounded to ±center_offset_max·R_prior, for
    #    profiles whose peak sits off the (refined) centreline.  Default off.
    fit_center_offset: bool = False,
    center_offset_max: float = 0.5,
):
    """Return characteristic (Q, V_CL, R) for a vessel via per-slice fits.

    Parameters
    ----------
    flow_avg : (H, W, 2) mean flow field, OpenCV [dx, dy]
    cl : (N, 2) refined centerline in [row, col] pixels
    R_prior : float — nominal half-width for sampling
    slice_window : int — averaging window along arc (default 2 = 5 pts)
    n_bins : int — number of ρ bins over ``[-R_prior, R_prior]``
    fit_fraction : float — fit only ``|ρ| <= fit_fraction × R_prior``
    outlier_mode : "sign" (same direction as the median; production), "median_band" (Kloosterman) or "iqr"
    median_lo, median_hi : accepted band as fractions of the |Q| median
    R_slice_lo, R_slice_hi : allowed fit-R range as fractions of R_prior
    use_r_prior_area : if True, compute ``Q_slice = ½ V_CL π R_prior²``;
        keeps the fit's R for the R-range sanity gate but uses mask R
        for area (kills the ~14–72% Q bloat we observed).

    Returns
    -------
    dict with:
      * ``Q_char, V_CL_char, R_char`` — characteristic scalars
      * ``Q_char_valid_frac`` — fraction of slices that survived
      * ``n_slices, n_valid`` — counts
      * ``per_slice_Q, per_slice_R, per_slice_v_CL, per_slice_arc``
      * ``per_slice_reason`` — string per slice describing keep/reject
      * ``valid_mask`` — bool per slice
    """
    from scipy.optimize import curve_fit

    H, W = flow_avg.shape[:2]
    N = len(cl)
    if N < min_valid_slices or R_prior <= 0:
        return dict(
            Q_char=float("nan"), V_CL_char=float("nan"), R_char=float("nan"),
            Q_char_valid_frac=0.0, n_slices=N, n_valid=0,
            per_slice_Q=np.full(N, np.nan), per_slice_R=np.full(N, np.nan),
            per_slice_v_CL=np.full(N, np.nan),
            per_slice_arc=(np.concatenate([[0], np.cumsum(np.hypot(
                np.diff(cl[:, 0]), np.diff(cl[:, 1])))]) if N > 1
                          else np.zeros(N)),
            per_slice_reason=[f"cl_too_short(N={N})"] * N,
            valid_mask=np.zeros(N, dtype=bool),
            reason="short_or_zero_R",
        )

    _, tang, norm = _centerline_geometry(cl)

    # Perpendicular sample grid — same for every slice.  Extended to
    # ±sample_frac·R_prior so the extent detector can reach past the mask R.
    _sR = float(max(sample_frac, 1.0)) * R_prior
    bin_edges = np.linspace(-_sR, _sR, n_bins + 1)
    rho = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Sample OF axial velocity at each (CL point i, bin j)
    v_axial_grid = np.full((N, n_bins), np.nan)
    for i in range(N):
        row_c, col_c = cl[i]
        n_r, n_c = norm[i]
        t_r, t_c = tang[i]
        for j, r_val in enumerate(rho):
            rr = row_c + r_val * n_r
            cc = col_c + r_val * n_c
            if not (0 <= rr < H - 1 and 0 <= cc < W - 1):
                continue
            r0i, c0i = int(rr), int(cc)
            fr, fc = rr - r0i, cc - c0i
            vx = ((1 - fr) * (1 - fc) * flow_avg[r0i, c0i, 0]
                  + fr * (1 - fc) * flow_avg[r0i + 1, c0i, 0]
                  + (1 - fr) * fc * flow_avg[r0i, c0i + 1, 0]
                  + fr * fc * flow_avg[r0i + 1, c0i + 1, 0])
            vy = ((1 - fr) * (1 - fc) * flow_avg[r0i, c0i, 1]
                  + fr * (1 - fc) * flow_avg[r0i + 1, c0i, 1]
                  + (1 - fr) * fc * flow_avg[r0i, c0i + 1, 1]
                  + fr * fc * flow_avg[r0i + 1, c0i + 1, 1])
            v_axial_grid[i, j] = (vy * t_r + vx * t_c)  # dt normalisation is caller's
            if mask is not None:
                mr, mc = int(round(rr)), int(round(cc))
                if not (0 <= mr < mask.shape[0]
                        and 0 <= mc < mask.shape[1] and mask[mr, mc]):
                    v_axial_grid[i, j] = np.nan

    # Windowed profile per slice — average over ±slice_window CL points.
    # `nanmean` emits a RuntimeWarning ("Mean of empty slice") when a window
    # is fully NaN; that's expected here (windows near vessel ends whose
    # ridge crop clipped past the mask) and produces a NaN result, which
    # downstream code already handles. Suppress the noise.
    import warnings as _w
    profiles = np.full((N, n_bins), np.nan)
    with np.errstate(all="ignore"), _w.catch_warnings():
        _w.filterwarnings("ignore", category=RuntimeWarning,
                          message="Mean of empty slice")
        for i in range(N):
            lo, hi = max(0, i - slice_window), min(N, i + slice_window + 1)
            profiles[i] = np.nanmean(v_axial_grid[lo:hi], axis=0)

    per_slice_v_CL = np.full(N, np.nan)
    per_slice_R    = np.full(N, np.nan)
    per_slice_Q    = np.full(N, np.nan)
    per_slice_r2   = np.full(N, np.nan)   # parabolic goodness-of-fit per slice
    per_slice_reason = ["ok"] * N
    fit_mask_prior = np.abs(rho) <= R_prior * fit_fraction

    per_slice_R_edt = np.full(N, np.nan)
    if fit_bound == "mask":
        if edt is None:
            raise ValueError("fit_bound='mask' requires edt array")
        for i, (r_c, c_c) in enumerate(cl):
            r_i, c_i = int(round(r_c)), int(round(c_c))
            if 0 <= r_i < edt.shape[0] and 0 <= c_i < edt.shape[1]:
                per_slice_R_edt[i] = float(edt[r_i, c_i])

    def _poi(r, v0, Rf, r0=0.0):
        return v0 * np.maximum(0.0, 1.0 - ((r - r0) / Rf) ** 2)

    per_slice_offset = np.full(N, np.nan)

    for i in range(N):
        p = profiles[i]
        if fit_bound == "mask" and np.isfinite(per_slice_R_edt[i]) \
                and per_slice_R_edt[i] > 1:
            fit_region = np.abs(rho) <= per_slice_R_edt[i]
        else:
            fit_region = fit_mask_prior
        valid = np.isfinite(p) & fit_region
        # Data-driven radial extent (replaces a fixed inner-fraction clip,
        # which either threw away good parabolic data or reached into the wall
        # / background).  Walk OUTWARD from the profile peak on each side and
        # keep bins while |v| stays above v_incl_frac·v_peak, stopping at the
        # transition where the parabola meets the wall / noise floor.  The walk
        # is contiguous (with a one-bin noise grace) so a distant bright spike
        # — e.g. an adjacent vessel — can't rejoin the fit, and it is per-side
        # so cropped / asymmetric sections are handled.  ``fit_region`` is only
        # a generous safety cap now; arc-direction problems (flows merging or
        # abruptly changing direction near a junction) are left to the
        # downstream shape (R²) and sign gates to reject.
        if fit_center_offset and np.isfinite(p).any():
            # centre the geometric cap on the profile peak (within the offset bound)
            _pk = float(np.clip(rho[int(np.nanargmax(np.abs(p)))],
                                -center_offset_max * R_prior, center_offset_max * R_prior))
            fit_region = np.abs(rho - _pk) <= R_prior * fit_fraction
            valid = np.isfinite(p) & fit_region
        if v_incl_frac > 0 and np.isfinite(p).any():
            ip = int(np.nanargmax(np.abs(p)))
            v_pk = abs(p[ip])
            if v_pk > 1e-9:
                thr = v_incl_frac * v_pk
                keep = np.zeros(n_bins, dtype=bool)
                keep[ip] = True
                for step in (1, -1):
                    j = ip + step
                    misses = 0
                    while 0 <= j < n_bins and fit_region[j]:
                        if valid[j] and abs(p[j]) >= thr:
                            keep[j] = True
                            misses = 0
                        else:
                            misses += 1
                            if misses > 1:
                                break
                        j += step
                if keep.sum() >= 5:
                    valid = keep
        if valid.sum() < 5:
            per_slice_reason[i] = f"few_pts({int(valid.sum())})"
            continue
        peak = p[np.nanargmax(np.abs(p))]
        try:
            # Constrain the fitted R to the same [lo, hi]·R_prior band as the
            # post-fit gate, so the free (v0, R) fit can't rail to a huge R on
            # broad/flat profiles (which then inflates Q).
            fit_kw = dict(p0=[peak if abs(peak) > 1e-6 else 0.1, R_prior],
                           bounds=([-np.inf, R_prior * R_slice_lo],
                                    [np.inf, R_prior * R_slice_hi]),
                           maxfev=2000)
            if fit_center_offset:
                _om = center_offset_max * R_prior
                _r0 = float(np.clip(rho[np.nanargmax(np.abs(np.where(valid, p, np.nan)))], -_om, _om))
                fit_kw["p0"] = fit_kw["p0"] + [_r0]
                fit_kw["bounds"] = (fit_kw["bounds"][0] + [-_om], fit_kw["bounds"][1] + [_om])
            if robust_loss == "huber":
                fit_kw["method"] = "trf"
                fit_kw["loss"]   = "huber"
                fit_kw["f_scale"] = 1.0
            popt, _ = curve_fit(_poi, rho[valid], p[valid], **fit_kw)
            v_CL, R_slice = float(popt[0]), float(popt[1])
            r0_slice = float(popt[2]) if fit_center_offset else 0.0
        except Exception:
            per_slice_reason[i] = "curve_fit_fail"
            continue
        if not np.isfinite(v_CL) or not np.isfinite(R_slice):
            per_slice_reason[i] = "nan_fit"
            continue
        if R_slice <= R_slice_lo * R_prior:
            per_slice_reason[i] = f"R_low({R_slice:.1f}<{R_slice_lo * R_prior:.1f})"
            continue
        if R_slice >= R_slice_hi * R_prior:
            per_slice_reason[i] = f"R_hi({R_slice:.1f}>{R_slice_hi * R_prior:.1f})"
            continue
        # Vertex must exceed the mean of the fitted velocities — a valid
        # downward parabola peaks above its own average; if not, the fit is
        # degenerate/flat (thesis: "max fitted velocity should be larger than
        # the average of the used interpolated velocities").
        if require_v0_gt_mean:
            _mean_used = float(np.mean(np.abs(p[valid])))
            if abs(v_CL) <= _mean_used:
                per_slice_reason[i] = f"v0<mean({abs(v_CL):.1f}<={_mean_used:.1f})"
                continue
        # Parabolic goodness-of-fit (R²) — the direct "is this cross-section
        # actually parabolic?" test.  Catches END / JUNCTION effects by SHAPE
        # (merges, adjacent vessels, flow turning) independent of arc position:
        # a non-parabolic slice fits poorly and is dropped.
        _pv = p[valid]
        _ss_tot = float(np.sum((_pv - np.mean(_pv)) ** 2))
        _r2 = (1.0 - float(np.sum((_pv - _poi(rho[valid], v_CL, R_slice, r0_slice)) ** 2))
               / _ss_tot) if _ss_tot > 1e-12 else 0.0
        per_slice_r2[i] = _r2
        if min_r2 > 0 and _r2 < min_r2:
            per_slice_reason[i] = f"non_parabolic(R2={_r2:.2f}<{min_r2:.2f})"
            continue
        R_for_area = R_prior if use_r_prior_area else R_slice
        per_slice_v_CL[i] = v_CL
        per_slice_R[i]    = R_slice
        per_slice_offset[i] = r0_slice
        per_slice_Q[i]    = 0.5 * v_CL * np.pi * R_for_area ** 2

    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cl[:, 0]),
                                                    np.diff(cl[:, 1])))])

    q_finite = np.isfinite(per_slice_Q)
    if q_finite.sum() < min_valid_slices:
        return dict(
            Q_char=float("nan"), V_CL_char=float("nan"), R_char=float("nan"),
            Q_char_valid_frac=0.0, n_slices=int(N), n_valid=int(q_finite.sum()),
            per_slice_Q=per_slice_Q, per_slice_R=per_slice_R,
            per_slice_v_CL=per_slice_v_CL, per_slice_arc=arc,
            per_slice_reason=per_slice_reason, valid_mask=q_finite,
            per_slice_r2=per_slice_r2, per_slice_profiles=profiles, rho=rho,
            reason="too_few_finite_slice_fits",
        )

    Q_med = float(np.nanmedian(per_slice_Q))
    if outlier_mode == "sign":
        # Production since 2026-11-09: no flow-rate band — every shape-valid section whose flow direction agrees with
        # the median is averaged (the 0.5–1.5× band cut good sections and slightly worsened repeat agreement).
        valid_mask = q_finite & (np.sign(per_slice_Q) == np.sign(Q_med))
        for i in range(N):
            if per_slice_reason[i] == "ok" and q_finite[i] and np.sign(per_slice_Q[i]) != np.sign(Q_med):
                per_slice_reason[i] = "sign_flip"
    elif outlier_mode == "median_band":
        lo_thr = median_lo * abs(Q_med)
        hi_thr = median_hi * abs(Q_med)
        valid_mask = (q_finite
                      & (np.abs(per_slice_Q) >= lo_thr)
                      & (np.abs(per_slice_Q) <= hi_thr)
                      & (np.sign(per_slice_Q) == np.sign(Q_med)))
        for i in range(N):
            if per_slice_reason[i] != "ok":
                continue
            if not q_finite[i]:
                per_slice_reason[i] = "Q_nan"; continue
            if np.sign(per_slice_Q[i]) != np.sign(Q_med):
                per_slice_reason[i] = "sign_flip"; continue
            aq = abs(per_slice_Q[i])
            if aq < lo_thr:
                per_slice_reason[i] = f"Q_lo({aq:.4f}<{lo_thr:.4f})"; continue
            if aq > hi_thr:
                per_slice_reason[i] = f"Q_hi({aq:.4f}>{hi_thr:.4f})"; continue
    else:
        q_vals = per_slice_Q[q_finite]
        q1, q3 = np.percentile(q_vals, [25, 75])
        iqr = q3 - q1
        lo_thr = q1 - iqr_k * iqr
        hi_thr = q3 + iqr_k * iqr
        valid_mask = (q_finite
                      & (per_slice_Q >= lo_thr) & (per_slice_Q <= hi_thr))
        for i in range(N):
            if per_slice_reason[i] != "ok":
                continue
            if not q_finite[i]:
                per_slice_reason[i] = "Q_nan"; continue
            if per_slice_Q[i] < lo_thr:
                per_slice_reason[i] = f"Q_lo({per_slice_Q[i]:.4f}<{lo_thr:.4f})"; continue
            if per_slice_Q[i] > hi_thr:
                per_slice_reason[i] = f"Q_hi({per_slice_Q[i]:.4f}>{hi_thr:.4f})"; continue

    if valid_mask.sum() < min_valid_slices:
        # Fewer than min_valid_slices sections survive the outlier rule (e.g. two fits of opposite direction): no flux.
        # Before 2026-11-11 this fell back to averaging ALL finite sections — contradictory signs included (102 of 14 396
        # 21S recordings, 74 % grossly off their repeats).
        for i in range(N):
            if per_slice_reason[i] == "ok" and not valid_mask[i]:
                per_slice_reason[i] = "outlier_rejected"
        return dict(
            Q_char=float("nan"), V_CL_char=float("nan"), R_char=float("nan"),
            Q_char_valid_frac=0.0, n_slices=int(N), n_valid=int(valid_mask.sum()),
            per_slice_Q=per_slice_Q, per_slice_R=per_slice_R,
            per_slice_v_CL=per_slice_v_CL, per_slice_arc=arc,
            per_slice_reason=per_slice_reason, valid_mask=valid_mask,
            per_slice_r2=per_slice_r2, per_slice_profiles=profiles, rho=rho,
            reason="too_few_consistent_slices",
        )

    Q_char    = float(np.nanmean(per_slice_Q[valid_mask]))
    V_CL_char = float(np.nanmean(per_slice_v_CL[valid_mask]))
    if abs(V_CL_char) > 1e-9:
        R_char = float(np.sqrt(2 * abs(Q_char) / (np.pi * abs(V_CL_char))))
    else:
        R_char = float("nan")

    return dict(
        Q_char=Q_char, V_CL_char=V_CL_char, R_char=R_char,
        Q_char_valid_frac=float(valid_mask.sum() / max(N, 1)),
        n_slices=int(N), n_valid=int(valid_mask.sum()),
        per_slice_Q=per_slice_Q, per_slice_R=per_slice_R,
        per_slice_v_CL=per_slice_v_CL, per_slice_arc=arc,
        per_slice_reason=per_slice_reason, per_slice_r2=per_slice_r2,
        # measured windowed cross-section profiles + the ρ grid, so a
        # walkthrough panel can draw the per-slice parabolas + measured points.
        per_slice_profiles=profiles, rho=rho, per_slice_offset=per_slice_offset,
        valid_mask=valid_mask, reason="ok",
    )
