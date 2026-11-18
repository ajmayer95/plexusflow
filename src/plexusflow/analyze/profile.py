"""Full-vessel Farneback OF pipeline: video → cross-section velocity
profile → Q (direct integration + Poiseuille fit) + Kloosterman scalar.

Trimmed from PerTileFlow's ``pertile.analysis.of_profile`` — kept only
the ``compute_of_profile_and_Q`` main entry, ``_rebin_profile`` helper,
and their live-dependencies (background normalisation, vessel-pixel
mask builder, flow-derived tangent).  Dropped the legacy centerline-
refinement modes (moved to ``centerline.py`` with only parabola_peak
kept), the xcorr-symmetry variants, and the PIV cross-correlation
branch.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from .centerline import _centerline_geometry, _refine_centerline
from .kloosterman import kloosterman_scalar


# ── Preprocessing helpers ─────────────────────────────────────────────

def _choose_dt(frames_u8, centerline, fb_kwargs, dt_max, target_disp_px, n_pilot=120, peak_factor=2.0):
    """Adaptive OF frame spacing for one vessel: (dt, d_peak).

    Speed comes from the TIME-AVERAGED dt=1 flow VECTOR at each centreline pixel (median over the centreline):
    Farneback noise at dt=1 is ~1 px in magnitude but has no preferred direction, so it averages out of the
    vector mean while real flow does not.  (Per-pair magnitudes read the noise floor as speed and kept slow
    vessels at dt=1.)  Peak displacement ≈ peak_factor × mean speed; dt = clip(round(target / d_peak), 1, dt_max)."""
    import cv2
    T, H, W = frames_u8.shape
    if T < 3:
        return 1, float("nan")
    rr = np.clip(np.round(centerline[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(centerline[:, 1]).astype(int), 0, W - 1)
    vsum = np.zeros((len(rr), 2)); n = 0
    for t in np.unique(np.linspace(0, T - 2, min(n_pilot, T - 1)).astype(int)):
        fl = cv2.calcOpticalFlowFarneback(frames_u8[t], frames_u8[t + 1], None, **fb_kwargs)
        vsum += fl[rr, cc, :2]; n += 1
    v_mean = float(np.median(np.hypot(*(vsum / max(n, 1)).T)))
    d_peak = peak_factor * v_mean
    if not np.isfinite(d_peak) or d_peak <= 0:
        return int(dt_max), d_peak
    return int(np.clip(round(target_disp_px / d_peak), 1, int(dt_max))), d_peak


def _normalise_for_of(frames: np.ndarray, bg_percentile: int = 10) -> np.ndarray:
    """Background-subtract float frames + rescale to uint8 for Farneback.

    If ``frames`` is already uint8, returned as-is.  For float input the
    background is a low percentile per pixel (keeps steady-flow signal
    that the temporal mean would erase).
    """
    if frames.dtype == np.uint8:
        return frames
    if frames.size == 0 or frames.shape[0] == 0:
        return np.zeros_like(frames, dtype=np.uint8)
    bg = np.percentile(frames, bg_percentile, axis=0)  # (H, W)
    frames_bt = frames - bg
    vmin = float(np.percentile(frames_bt, 1))
    vmax = float(np.percentile(frames_bt, 99))
    frames_norm = np.clip((frames_bt - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    return (frames_norm * 255).astype(np.uint8)


def _build_vessel_mask(centerline: np.ndarray, R_px: float, H: int, W: int) -> dict:
    """Enumerate vessel pixels within R of the centerline (excluding endpoint caps).

    Returns dict with ``rr_in``, ``cc_in`` (int arrays of pixel indices),
    ``rho_in`` (signed perpendicular distance), ``t_row_in``/``t_col_in``
    (local tangent components), ``n_row_in``/``n_col_in`` (normals),
    ``nearest`` (index of nearest CL point per pixel).
    """
    _, tangents_cl, normals_cl = _centerline_geometry(centerline)
    cl_tree = cKDTree(centerline)
    rr_g, cc_g = np.mgrid[0:H, 0:W]
    pts_g = np.column_stack([rr_g.ravel().astype(float),
                             cc_g.ravel().astype(float)])
    dist_g, nearest_g = cl_tree.query(pts_g)
    mask = ((dist_g.reshape(H, W) <= R_px)
            & (nearest_g.reshape(H, W) > 0)
            & (nearest_g.reshape(H, W) < len(centerline) - 1))
    rr_in, cc_in = np.where(mask)
    pts_in = np.column_stack([rr_in.astype(float), cc_in.astype(float)])
    _, nearest = cl_tree.query(pts_in)
    t_row_in = tangents_cl[nearest, 0]
    t_col_in = tangents_cl[nearest, 1]
    n_row_in = normals_cl[nearest, 0]
    n_col_in = normals_cl[nearest, 1]
    rho_in = ((rr_in - centerline[nearest, 0]) * n_row_in
              + (cc_in - centerline[nearest, 1]) * n_col_in)
    return dict(rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
                t_row_in=t_row_in, t_col_in=t_col_in,
                n_row_in=n_row_in, n_col_in=n_col_in,
                nearest=nearest)


def _compute_flow_tangent(
    flow_avg, rr_in, cc_in, nearest, tang_geom, centerline,
    tangent_smooth_sigma: float = 5.0, dt: float = 1,
):
    """Derive per-pixel tangent from the mean flow field, smoothed along arc.

    Returns ``(t_row, t_col)`` arrays of the same length as ``rr_in``.
    Falls back to the geometric tangent where the OF magnitude is
    below ~0.03 px/frame.
    """
    flow_vr = flow_avg[rr_in, cc_in, 1]
    flow_vc = flow_avg[rr_in, cc_in, 0]
    flow_tang = np.column_stack([flow_vr, flow_vc])
    flow_mag = np.sqrt(flow_tang[:, 0] ** 2 + flow_tang[:, 1] ** 2)
    flow_mag_safe = np.maximum(flow_mag, 1e-8)
    flow_tang /= flow_mag_safe[:, None]
    cl_tang = tang_geom[nearest]
    dot = (flow_tang * cl_tang).sum(axis=1)
    flow_tang[dot < 0] *= -1
    weak = flow_mag < 0.03
    flow_tang[weak] = cl_tang[weak]
    n_cl = len(centerline)
    tang_cl_sum = np.zeros((n_cl, 2))
    tang_cl_w = np.zeros(n_cl)
    for k in range(len(nearest)):
        j = nearest[k]
        w = float(flow_mag[k])
        tang_cl_sum[j] += w * flow_tang[k]
        tang_cl_w[j] += w
    has_data = tang_cl_w > 1e-8
    tang_cl_avg = np.copy(tang_geom)
    tang_cl_avg[has_data] = tang_cl_sum[has_data] / tang_cl_w[has_data, None]
    if tangent_smooth_sigma > 0 and n_cl > 3:
        tang_cl_avg[:, 0] = gaussian_filter1d(tang_cl_avg[:, 0], tangent_smooth_sigma)
        tang_cl_avg[:, 1] = gaussian_filter1d(tang_cl_avg[:, 1], tangent_smooth_sigma)
        nrm = np.sqrt((tang_cl_avg ** 2).sum(axis=1, keepdims=True))
        tang_cl_avg /= np.maximum(nrm, 1e-8)
    flow_tang_smooth = tang_cl_avg[nearest]
    return flow_tang_smooth[:, 0], flow_tang_smooth[:, 1]


def _central_vessel_extent(bin_centers, profile_v, R,
                           smooth_bins=1.5, rise_frac=0.4, dip_frac=0.25):
    """Boolean mask of bins belonging to the CENTRAL vessel.

    A single Poiseuille vessel's ``|v|`` decreases monotonically from the
    centreline peak toward ~0 at the wall.  When the sampling ROI bleeds into
    an adjacent vessel the aggregate profile dips to a *valley* and then climbs
    again — every bin past that valley is a different vessel and would drag a
    free-``R`` fit far too wide (see the taper≈0.4, R_free/R_seg≈2.4 case).

    Detection: anchor the central peak near the centreline (largest ``|v|``
    within ``±0.5R`` of ρ=0 — the core, not a taller wing bump), then walk
    outward each direction tracking the running minimum.  When the
    profile climbs back above ``valley + rise_frac·(peak − valley)`` *after* a
    real dip (depth ≥ ``dip_frac·peak``), cut at the valley; exclude everything
    beyond it.  A clean vessel (monotone decay to ~0, only noise wiggles at the
    edge) never triggers, so this is safe to apply unconditionally.

    Returns a boolean mask over ``bin_centers`` (True = keep for the free-R fit).
    """
    v = np.asarray(profile_v, dtype=float)
    bc = np.asarray(bin_centers, dtype=float)
    n = len(v)
    good = np.isfinite(v)
    if good.sum() < 3:
        return good.copy()
    # Interpolate NaNs, take |v|, lightly smooth for robust valley detection.
    idx = np.arange(n)
    vv = v.copy()
    if (~good).any():
        vv[~good] = np.interp(idx[~good], idx[good], v[good])
    vs = gaussian_filter1d(np.abs(vv), max(smooth_bins, 0.5))

    # Anchor the peak NEAR the centreline (ρ=0), where the vessel core is by
    # definition — NOT at the global max within ±R.  On a bumpy/large structure
    # the tallest bin can sit on a wing (e.g. R=52µm vessel with a +40µm bump
    # taller than the ρ=0 core); walking outward from that wing then rejects the
    # real core as "neighbour".  Search only within ±0.5R of ρ=0 for the anchor.
    _dbc = (bc[1] - bc[0]) if n > 1 else R
    near0 = np.where(np.abs(bc) <= max(0.5 * R, _dbc))[0]
    if len(near0) == 0:
        near0 = np.where(np.abs(bc) <= max(R, _dbc))[0]
    if len(near0) == 0:
        near0 = idx
    peak = int(near0[np.argmax(vs[near0])])
    amp = float(vs[peak])
    if amp <= 0:
        return good.copy()

    def _boundary(step):
        run_min = vs[peak]
        run_min_i = peak
        i = peak + step
        while 0 <= i < n:
            if vs[i] < run_min:
                run_min = vs[i]
                run_min_i = i
            elif (amp - run_min) >= dip_frac * amp and \
                    vs[i] > run_min + rise_frac * (amp - run_min):
                return run_min_i          # valley: neighbour begins past here
            i += step
        return (i - step)                 # reached the end, no neighbour

    lo = _boundary(-1)
    hi = _boundary(+1)
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi + 1] = True
    return mask & good


# ── Main entry ─────────────────────────────────────────────────────────

def compute_of_profile_and_Q(
    frames,                    # (T, H, W) uint8 or float
    centerline,                # (N, 2) float [row, col]
    vessel_radius_px,          # float — vessel radius in pixels
    *,
    # Background subtraction
    bg_percentile: int = 10,
    # Farneback params
    fb_winsize: int = 5,
    fb_levels: int = 1,
    fb_iterations: int = 3,
    fb_poly_n: int = 5,
    fb_poly_sigma: float = 1.1,
    fb_pyr_scale: float = 0.5,
    dt: Union[int, str] = 1,
    # dt="auto": per-vessel frame spacing.  A pilot of ~40 dt=1 pairs measures
    # the vessel's peak centreline displacement d95 (px/frame); the run then
    # uses dt = clip(round(target_disp_px / d95), 1, dt_max), so fast vessels
    # stay at dt=1 (no aliasing through Farneback's ~1–2 px capture range) and
    # very slow ones get enough displacement to rise above the OF noise floor
    # (at dt=1 sub-pixel noise is rectified into spurious flow).
    dt_max: int = 12,
    target_disp_px: float = 1.0,
    # Profile
    n_bins: int = 30,
    fit_fraction: float = 0.75,
    # Sample the OF / bin the profile out to mask_frac·R (not just R): the
    # segmented mask radius is only an initial guess, and if the centerline is
    # offset a tight ±R window clips off part of the real vessel and hides the
    # true extent.  Fits still reference the segmented R; the wider data lets
    # the free-R fit and the per-slice extent detector see where flow actually
    # ends instead of extrapolating (railing).
    mask_frac: float = 2.0,
    v_min_threshold: float = 0.03,
    # Resample the centerline to this fixed arc spacing (px) before anything,
    # so the number of per-slice Kloosterman cross-sections scales with vessel
    # LENGTH instead of the arbitrary graph-path point density.  Short vessels
    # were getting only a handful of slices (whatever the graph stored) and
    # failing the R² gate.  0 disables.
    cl_resample_px: float = 1.0,
    # Centreline refinement
    refine_centerline: bool = False,
    ridge_mask=None,
    ridge_iterations: int = 1,
    ridge_shift_tol_px: float = 1.0,
    ridge_offset_smooth_sigma: float = 4.0,
    # Tangent
    use_flow_tangent: bool = True,
    tangent_smooth_sigma: float = 5.0,
    # Kloosterman: area from the per-slice FITTED radius (False) vs the mask
    # R_prior (True).  Default False = fitted radius (the defensible choice).
    kl_use_r_prior_area: bool = False,
    # Per-slice fitted-R sanity bounds as fractions of R_prior.  Kept LOOSE
    # (thesis: "fitted diameter ≤ 2× the cross-section linear dimension"); a fit
    # that rails at a bound is rejected (R_low / R_hi) — in practice none do
    # (2026-11-09: 0 % of rejections), shape (R²) failures dominate.
    kl_r_slice_lo: float = 0.5,
    kl_r_slice_hi: float = 2.0,
    # Velocity-relative inclusion threshold (fraction of centerline velocity)
    # for cross-section points — adaptive edge/mask-artifact rejection.
    kl_v_incl_frac: float = 0.25,
    # Generous geometric safety cap for the Kloosterman per-slice fit window
    # (fraction of R_prior).  The velocity-transition walk is the real limiter;
    # this just stops the search from running past the sampled half-width.
    kl_fit_fraction: float = 1.0,
    # Reject a slice whose measured cross-section isn't actually parabolic
    # (goodness-of-fit R² below this) — detects end/junction effects by SHAPE.
    kl_min_r2: float = 0.5,
    # Cardiac phase (used to prefer systolic frames during CL refinement)
    f0_hz: Optional[float] = None,
    frame_dt_s: Optional[float] = None,
    return_diagnostics: bool = False,
    roi_margin_px: Optional[int] = None,
    # Legacy parameters accepted but unused (only parabola_peak refinement
    # is supported in the plexusflow port).  Accepted so callers built for
    # the old pertile API don't break.
    refinement_mode: str = "parabola_peak",  # noqa: ARG001
    ridge_ang_tol_deg: float = 35.0,          # noqa: ARG001
    ridge_prox_sigma=None,                    # noqa: ARG001
    max_shift: float = 10.0,                  # noqa: ARG001
    n_shift_steps: int = 21,                  # noqa: ARG001
):
    """Full Farneback OF pipeline on a vessel segment.

    Runs OF pair-by-pair, projects onto the local tangent to get axial
    velocity per pixel, accumulates a ρ-binned cross-section profile,
    fits Poiseuille (both R-fixed and R-free), does the direct-integration
    plug Q, and runs the Kloosterman per-slice scalar.

    Returns a dict with the same keys the pertile version returned
    (backwards-compatible), including the four Q variants:
      * ``Q_plug_mean``     — direct pixel-integration of v_axial × π R²
      * ``Q_pois_mean``     — R-fixed Poiseuille ``v0_fixed × π R² / 2``
      * ``Q_kloosterman``   — per-slice Poiseuille, shape-valid sign-matched sections averaged
      * plus the raw fit params: ``v0_fixed``, ``v0_free``, ``R_fit_free``,
        etc. — from which ``Q_pois_free = v0_free × π R_free² / 2`` can
        be computed downstream.
    """
    import cv2

    frames = np.asarray(frames)
    centerline = np.asarray(centerline, dtype=float)
    # Resample to uniform ~cl_resample_px arc spacing so slice count scales with
    # vessel length (never fewer points than the input; capped for runtime).
    if cl_resample_px and cl_resample_px > 0 and len(centerline) >= 2:
        _d = np.hypot(np.diff(centerline[:, 0]), np.diff(centerline[:, 1]))
        _s = np.concatenate([[0.0], np.cumsum(_d)])
        _tot = float(_s[-1])
        if _tot >= 2 * cl_resample_px:
            _n = int(np.clip(round(_tot / cl_resample_px) + 1,
                             max(3, len(centerline)), 150))
            _sn = np.linspace(0.0, _tot, _n)
            centerline = np.column_stack([np.interp(_sn, _s, centerline[:, 0]),
                                          np.interp(_sn, _s, centerline[:, 1])])
    R = float(vessel_radius_px)                 # segmented radius (physical)
    R_samp = R * float(max(mask_frac, 1.0))     # OF-sampling / mask half-width
    T, H_full, W_full = frames.shape
    fb_winsize = int(fb_winsize) | 1     # force odd

    # ── 0. Crop to ROI around centreline ─────────────────────────────
    margin = (int(np.ceil(R * 2)) + 4 if roi_margin_px is None
              else max(int(roi_margin_px), int(np.ceil(R_samp)) + 2))
    r0 = max(0, int(np.floor(centerline[:, 0].min())) - margin)
    r1 = min(H_full, int(np.ceil(centerline[:, 0].max())) + margin + 1)
    c0 = max(0, int(np.floor(centerline[:, 1].min())) - margin)
    c1 = min(W_full, int(np.ceil(centerline[:, 1].max())) + margin + 1)
    frames = frames[:, r0:r1, c0:c1]
    centerline = centerline - np.array([[r0, c0]], dtype=float)
    cl_initial_roi = centerline.copy()   # ROI-local initial CL (for the walkthrough)
    T, H, W = frames.shape
    if ridge_mask is not None:
        rm = np.asarray(ridge_mask)
        if rm.shape[:2] != (H_full, W_full):
            _ridge_mask_roi = rm.astype(bool)
        else:
            _ridge_mask_roi = rm[r0:r1, c0:c1].astype(bool)
    else:
        _ridge_mask_roi = None

    fb_kwargs = dict(
        pyr_scale=fb_pyr_scale, levels=fb_levels,
        winsize=fb_winsize, iterations=fb_iterations,
        poly_n=fb_poly_n, poly_sigma=fb_poly_sigma, flags=0,
    )

    # ── 1. Background subtraction → uint8 for Farneback ──────────────
    if frames.dtype != np.uint8 and bg_percentile > 0:
        frames_u8 = _normalise_for_of(frames, bg_percentile)
    elif frames.dtype == np.uint8:
        frames_u8 = frames
    else:
        vmin = float(np.percentile(frames, 1))
        vmax = float(np.percentile(frames, 99))
        frames_u8 = np.clip((frames - vmin) / max(vmax - vmin, 1e-8) * 255,
                             0, 255).astype(np.uint8)

    # ── 1b. Adaptive frame spacing ───────────────────────────────────
    of_pilot_disp = float("nan")
    if isinstance(dt, str):
        dt, of_pilot_disp = _choose_dt(frames_u8, centerline, fb_kwargs, dt_max, target_disp_px)

    # ── 2. Centreline geometry / optional refinement ────────────────
    cl_shift = 0.0
    cl_used = centerline.copy()
    # Refinement intermediates for the walkthrough figure (additive; only
    # populated when return_diagnostics, but cheap to always fill).
    _refine_diag = dict(
        cl_initial=cl_initial_roi, cl_coarse=cl_initial_roi,
        coarse_shift=0.0, offsets_raw=None, offsets_smooth=None,
        flow_avg=None, refine_mask=None)
    if refine_centerline:
        geom = _refine_centerline(
            frames_u8, centerline, R, fb_kwargs, dt=dt,
            f0_hz=f0_hz, frame_dt_s=frame_dt_s,
            tangent_smooth_sigma=tangent_smooth_sigma,
            use_flow_tangent=use_flow_tangent,
            ridge_mask=_ridge_mask_roi,
            ridge_iterations=ridge_iterations,
            ridge_shift_tol_px=ridge_shift_tol_px,
            ridge_offset_smooth_sigma=ridge_offset_smooth_sigma,
            mask_frac=mask_frac,
        )
        cl_used = geom["cl_refined"]
        cl_shift = geom["shift_applied"]
        rr_in = geom["rr_in"]
        cc_in = geom["cc_in"]
        rho_in = geom["rho_in"]
        t_row_in = geom["t_row_in"]
        t_col_in = geom["t_col_in"]
        n_vessel_px = geom["n_vessel_px"]
        _refine_diag.update(
            cl_initial=geom.get("cl_initial", cl_initial_roi),
            cl_coarse=geom.get("cl_coarse", cl_initial_roi),
            coarse_shift=geom.get("coarse_shift", 0.0),
            offsets_raw=geom.get("offsets_raw"),
            offsets_smooth=geom.get("offsets_smooth"),
            flow_avg=geom.get("flow_avg"),
            refine_mask=geom.get("mask"),
            reverted=geom.get("reverted", False),
            refine_peak_prior=geom.get("refine_peak_prior"),
            refine_peak_ref=geom.get("refine_peak_ref"))
    else:
        geom = _build_vessel_mask(cl_used, R_samp, H, W)
        rr_in = geom["rr_in"]
        cc_in = geom["cc_in"]
        rho_in = geom["rho_in"]
        t_row_in = geom["t_row_in"]
        t_col_in = geom["t_col_in"]
        nearest = geom["nearest"]
        n_vessel_px = int(len(rr_in))
        # Optionally replace geometric tangent with flow-derived one
        if use_flow_tangent:
            n_warmup = min(80, T - dt)
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
            _, tang_geom, _ = _centerline_geometry(cl_used)
            t_row_in, t_col_in = _compute_flow_tangent(
                flow_avg, rr_in, cc_in, nearest, tang_geom, cl_used,
                tangent_smooth_sigma=tangent_smooth_sigma, dt=dt,
            )
            _refine_diag["flow_avg"] = flow_avg

    if n_vessel_px < 10:
        nan = float("nan")
        return dict(
            Q_plug_t=np.array([]), Q_pois_t=np.array([]),
            Q_plug_mean=nan, Q_pois_mean=nan,
            v0_fixed=nan, v0_free=nan, R_fit_free=R,
            bin_centers=np.array([]), profile_v=np.array([]),
            cl_used=cl_used, cl_shift=cl_shift,
            rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
            t_row_in=t_row_in, t_col_in=t_col_in,
            vfield_mean=np.zeros((H, W, 2)),
            n_vessel_px=n_vessel_px, fb_winsize=fb_winsize,
            roi_offset=(r0, c0),
            Q_kloosterman=nan, R_kloosterman=nan, V_CL_kloosterman=nan,
            Q_kloosterman_valid_frac=0.0, Q_kloosterman_n_valid=0,
        )

    # ── 3. Per-frame OF loop ─────────────────────────────────────────
    A_px2 = np.pi * R ** 2
    n_pairs = T - dt

    # Bin the profile out to ±R_samp (wider than the mask R) at ~the same bin
    # width, so we see the flow beyond the segmented radius.
    n_bins_w = max(n_bins, int(round(n_bins * R_samp / max(R, 1e-6))))
    bin_edges = np.linspace(-R_samp, R_samp, n_bins_w + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_members = [(rho_in >= lo) & (rho_in < hi)
                   for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    n_bins = n_bins_w

    profile_sum = np.zeros(n_bins)
    profile_cnt = np.zeros(n_bins)
    Q_plug_t = np.full(n_pairs, np.nan)
    per_frame_v0_fixed = np.full(n_pairs, np.nan)

    vfield_sum = np.zeros((H, W, 2), dtype=np.float64)
    vfield_cnt = np.zeros((H, W), dtype=np.float64)
    flow_in_t = np.zeros((n_pairs, len(rr_in), 2), dtype=np.float32)   # per-frame field on the vessel pixels (Q_kl_t)

    r_cut = R * fit_fraction
    _diag_t = (n_pairs // 2) if (return_diagnostics and n_pairs > 0) else -1
    flow_sample = None

    for t in range(n_pairs):
        flow = cv2.calcOpticalFlowFarneback(
            frames_u8[t], frames_u8[t + dt], None, **fb_kwargs)
        if t == _diag_t:
            flow_sample = flow.copy()

        of_row = flow[rr_in, cc_in, 1] / dt
        of_col = flow[rr_in, cc_in, 0] / dt
        v_axial = of_row * t_row_in + of_col * t_col_in

        flow_in_t[t, :, 0] = flow[rr_in, cc_in, 0] / dt
        flow_in_t[t, :, 1] = flow[rr_in, cc_in, 1] / dt
        vfield_sum[rr_in, cc_in, 0] += flow[rr_in, cc_in, 0] / dt
        vfield_sum[rr_in, cc_in, 1] += flow[rr_in, cc_in, 1] / dt
        vfield_cnt[rr_in, cc_in] += 1.0

        of_mag = np.sqrt(of_row ** 2 + of_col ** 2)
        active = of_mag >= v_min_threshold

        if active.any():
            Q_plug_t[t] = float(np.nanmean(v_axial[active])) * A_px2

        for i_bin, in_bin in enumerate(bin_members):
            m = in_bin & active
            vals = v_axial[m]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                profile_sum[i_bin] += float(np.mean(finite)) * int(m.sum())
                profile_cnt[i_bin] += int(m.sum())

        # Per-frame R-fixed Poiseuille fit.  Every vessel pixel in every frame — NO per-frame speed threshold: the pooled
        # amplitude v0(t) is proportional to the common waveform only if the pixel weights are fixed in time; a threshold
        # drops slow near-wall pixels at the trough, inflating it and flattening the pulsatility (known-truth benchmark:
        # |H1| recovery 0.930 → 0.947, slowest vessels 0.93 → 0.97; tile 36 +0.5 %; 2026-11-13).
        r_valid_t, v_valid_t = [], []
        for i_bin, in_bin in enumerate(bin_members):
            if abs(bin_centers[i_bin]) > r_cut:
                continue
            m = in_bin
            vals = v_axial[m]
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2:
                r_valid_t.append(bin_centers[i_bin])
                v_valid_t.append(float(np.mean(finite)))
        if len(r_valid_t) >= 3:
            rv_t = np.array(r_valid_t)
            vv_t = np.array(v_valid_t)
            A_mat = (1.0 - (rv_t / R) ** 2)[:, None]
            try:
                v0, *_ = np.linalg.lstsq(A_mat, vv_t, rcond=None)
                per_frame_v0_fixed[t] = float(v0[0])
            except Exception:
                pass

    # ── 4. Time-averaged profile + velocity field ────────────────────
    with np.errstate(invalid="ignore"):
        profile_v = np.where(profile_cnt > 0, profile_sum / profile_cnt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        vfield_mean = np.where(
            vfield_cnt[..., None] > 0,
            vfield_sum / vfield_cnt[..., None], 0.0)

    # ── 5. Time-averaged Poiseuille fits (R-fixed + R-free) ──────────
    valid = np.isfinite(profile_v)
    v0_fixed = np.nan
    v0_free = np.nan
    R_fit_free = R
    # Neighbour rejection: when the wide ROI bleeds into an adjacent vessel the
    # profile dips to a valley and climbs again.  Cut at that valley so the
    # free-R fit sees only the central vessel (mask-R is already inside ±R).
    _central = _central_vessel_extent(bin_centers, profile_v, R)
    _neighbour = valid & ~_central     # bins thrown out as a different vessel
    if valid.sum() >= 3:
        _free = valid & _central
        if _free.sum() < 3:            # degenerate cut → fall back to all valid
            _free = valid
        rv = bin_centers[_free]        # central-vessel data only (free-R fit)
        vv = profile_v[_free]
        # R-fixed (mask-R) fit uses only the inner segmented radius — the wider
        # bins beyond R would drive the 1-(r/R)² model negative and corrupt it.
        _mfit = valid & (np.abs(bin_centers) <= R * fit_fraction)
        if _mfit.sum() >= 3:
            rvm = bin_centers[_mfit]; vvm = profile_v[_mfit]
            try:
                coeffs, *_ = np.linalg.lstsq(
                    (1.0 - (rvm / R) ** 2)[:, None], vvm, rcond=None)
                v0_fixed = float(coeffs[0])
            except Exception:
                pass

        def pois_model(r, v0, Rf):
            return v0 * np.maximum(0.0, 1.0 - (r / Rf) ** 2)

        try:
            if np.isfinite(v0_fixed):
                s = -1.0 if v0_fixed < 0 else 1.0
            else:
                s = -1.0 if abs(vv.min()) > abs(vv.max()) else 1.0
            vv_flip = s * vv
            popt, _ = curve_fit(
                pois_model, rv, vv_flip,
                p0=[max(vv_flip.max(), 1e-3), R],
                bounds=([0, R * 0.3], [np.inf, R * 3.0]),
                maxfev=2000)
            v0_free = s * float(popt[0])
            R_fit_free = float(popt[1])
        except Exception:
            pass

    # ── 6. Per-frame Q_pois_t + means ────────────────────────────────
    Q_pois_t = per_frame_v0_fixed * A_px2 / 2.0
    Q_plug_mean = float(np.nanmean(Q_plug_t))
    Q_pois_mean = float(np.nanmean(Q_pois_t))

    # ── 7. Kloosterman scalar (uses already-dt-normalised vfield_mean) ─
    try:
        _kl = kloosterman_scalar(
            vfield_mean, cl_used, R, dt=1,
            slice_window=2, n_bins=n_bins, fit_fraction=kl_fit_fraction,
            outlier_mode="sign",
            R_slice_lo=kl_r_slice_lo, R_slice_hi=kl_r_slice_hi,
            min_valid_slices=2,
            use_r_prior_area=kl_use_r_prior_area,
            v_incl_frac=kl_v_incl_frac,
            min_r2=kl_min_r2,
            sample_frac=mask_frac,
            mask=_ridge_mask_roi,
        )
    except Exception as _e:
        _kl = dict(Q_char=float("nan"), V_CL_char=float("nan"),
                    R_char=float("nan"), Q_char_valid_frac=0.0,
                    n_slices=0, n_valid=0, reason=f"err:{_e}")

    # ── 8. Per-frame flux from the same Kloosterman fit (2026-11-17) ─────────────────────────────────────────────────
    # The time-mean fit's geometry (accepted sections, fit bins, R_s, offsets) is frozen and only the centreline speed is
    # refitted per frame — a linear operator w on the field (kloosterman.kloosterman_operator).  vfield_mean is the plain
    # mean of the per-frame fields on the vessel pixels (zero elsewhere), so mean_t Q_kl_t = Q_kloosterman exactly; no
    # rescaling of a second estimator.
    Q_kl_t = np.array([])
    if np.isfinite(_kl.get("Q_char", np.nan)):
        try:
            from .kloosterman import kloosterman_operator
            _w = kloosterman_operator(_kl, cl_used, vfield_mean.shape, mask=_ridge_mask_roi, slice_window=2)
            if _w is not None:
                _w = _w.toarray().ravel(); _ix = (np.asarray(rr_in) * W + np.asarray(cc_in)) * 2
                Q_kl_t = flow_in_t[:, :, 0].astype(np.float64) @ _w[_ix] + flow_in_t[:, :, 1].astype(np.float64) @ _w[_ix + 1]
        except Exception:
            Q_kl_t = np.array([])

    return dict(
        Q_plug_t=Q_plug_t,
        Q_pois_t=Q_pois_t,
        Q_kl_t=Q_kl_t,
        Q_plug_mean=Q_plug_mean,
        Q_pois_mean=Q_pois_mean,
        v0_fixed=v0_fixed,
        v0_free=v0_free,
        R_fit_free=R_fit_free,
        bin_centers=bin_centers,
        profile_v=profile_v,
        cl_used=cl_used,
        cl_shift=cl_shift,
        rr_in=rr_in,
        cc_in=cc_in,
        rho_in=rho_in,
        t_row_in=t_row_in,
        t_col_in=t_col_in,
        vfield_mean=vfield_mean,
        n_vessel_px=n_vessel_px,
        fb_winsize=fb_winsize,
        roi_offset=(r0, c0),
        of_dt=int(dt), of_pilot_disp_px=of_pilot_disp,
        # Kloosterman per-cross-section scalar Q
        Q_kloosterman=_kl.get("Q_char", float("nan")),
        R_kloosterman=_kl.get("R_char", float("nan")),
        V_CL_kloosterman=_kl.get("V_CL_char", float("nan")),
        Q_kloosterman_valid_frac=_kl.get("Q_char_valid_frac", 0.0),
        Q_kloosterman_n_valid=_kl.get("n_valid", 0),
        Q_kloosterman_per_slice=_kl.get("per_slice_Q", np.array([])),
        R_kloosterman_per_slice=_kl.get("per_slice_R", np.array([])),
        R2_kloosterman_per_slice=_kl.get("per_slice_r2", np.array([])),
        # Time-resolved centreline velocity for downstream harmonic fits
        v0_fixed_t=per_frame_v0_fixed,
        R_used=R,
        n_pairs=n_pairs,
        diag=(dict(
            frame_raw=(np.asarray(frames[_diag_t]) if _diag_t >= 0 else None),
            frame_bg=(np.asarray(frames_u8[_diag_t]) if _diag_t >= 0 else None),
            flow_sample=flow_sample,
            frame_idx=_diag_t,
            # ── Full stage-by-stage intermediates for the walkthrough figure.
            roi_offset=(r0, c0), roi_shape=(H, W), margin_px=margin,
            R_prior_px=R, fit_fraction=fit_fraction, n_bins=n_bins,
            bin_edges=bin_edges,
            cl_initial=_refine_diag.get("cl_initial"),
            cl_coarse=_refine_diag.get("cl_coarse"),
            coarse_shift=_refine_diag.get("coarse_shift"),
            offsets_raw=_refine_diag.get("offsets_raw"),
            offsets_smooth=_refine_diag.get("offsets_smooth"),
            flow_avg=_refine_diag.get("flow_avg"),
            refine_mask=_refine_diag.get("refine_mask"),
            refine_reverted=_refine_diag.get("reverted", False),
            refine_peak_prior=_refine_diag.get("refine_peak_prior"),
            refine_peak_ref=_refine_diag.get("refine_peak_ref"),
            frame_mean=(np.asarray(frames).mean(axis=0)
                        if frames.shape[0] else None),
            tangent_smooth_sigma=tangent_smooth_sigma,
            ridge_offset_smooth_sigma=ridge_offset_smooth_sigma,
            bg_percentile=bg_percentile,
            # Farneback / smoothing params (for the field-smoothing figure).
            fb_winsize=fb_winsize, fb_poly_n=fb_poly_n,
            fb_poly_sigma=fb_poly_sigma, fb_levels=fb_levels,
            fb_iterations=fb_iterations, dt=dt, n_pairs=n_pairs,
            # Per-pixel axial-projection geometry (raw→smooth on a cross-section).
            rr_in=rr_in, cc_in=cc_in, rho_in=rho_in,
            t_row_in=t_row_in, t_col_in=t_col_in,
            kl_use_r_prior_area=kl_use_r_prior_area,
            kl_r_slice_lo=kl_r_slice_lo, kl_r_slice_hi=kl_r_slice_hi,
            kl_v_incl_frac=kl_v_incl_frac, kl_min_r2=kl_min_r2,
            # Neighbour-rejection: bins kept for the free-R fit vs thrown out as
            # a bleed-in adjacent vessel (for shading in the radial-profile panel).
            central_mask=_central,
            neighbour_mask=_neighbour,
            # Full Kloosterman per-slice breakdown (v_CL, arc, reason, mask).
            kl=_kl,
        ) if return_diagnostics else None),
    )


# ── Re-bin helper (for comparing before/after CL refinement) ─────────

def _rebin_profile(unified_result, original_centerline, R, *,
                   n_bins: int = 30, fit_fraction: float = 0.75):
    """Re-bin the time-averaged velocity field from an existing result
    using a different centerline (typically the original graph CL, to
    check whether refinement improved things).

    Uses ``vfield_mean`` from the input result — no extra Farneback pass.
    Returns profile + Q_plug/Q_pois means, but NOT per-frame Q(t).
    """
    vfield = unified_result.get("vfield_mean")
    roi_offset = unified_result.get("roi_offset", (0, 0))
    if vfield is None:
        return dict(bin_centers=np.zeros(n_bins), profile_v=np.full(n_bins, np.nan),
                    v0_fixed=np.nan, Q_plug_mean=np.nan, Q_pois_mean=np.nan,
                    Q_plug_t=None, Q_pois_t=None)

    r0_off, c0_off = roi_offset
    H_roi, W_roi = vfield.shape[:2]
    cl_roi = original_centerline.copy()
    cl_roi[:, 0] -= r0_off
    cl_roi[:, 1] -= c0_off

    geom = _build_vessel_mask(cl_roi, R, H_roi, W_roi)
    if geom is None:
        return dict(bin_centers=np.zeros(n_bins), profile_v=np.full(n_bins, np.nan),
                    v0_fixed=np.nan, Q_plug_mean=np.nan, Q_pois_mean=np.nan,
                    Q_plug_t=None, Q_pois_t=None)

    rr_in = geom["rr_in"]
    cc_in = geom["cc_in"]
    rho_in = geom["rho_in"]
    t_row_in = geom["t_row_in"]
    t_col_in = geom["t_col_in"]

    of_row = vfield[rr_in, cc_in, 1]
    of_col = vfield[rr_in, cc_in, 0]
    v_axial = of_row * t_row_in + of_col * t_col_in

    bin_edges = np.linspace(-R, R, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile_v = np.full(n_bins, np.nan)
    for i_bin, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (rho_in >= lo) & (rho_in < hi)
        vals = v_axial[in_bin]
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            profile_v[i_bin] = float(np.mean(finite))

    A_px2 = np.pi * R ** 2
    v_mean = float(np.nanmean(v_axial[np.isfinite(v_axial)])) if np.any(np.isfinite(v_axial)) else 0.0
    Q_plug_mean = v_mean * A_px2

    fit_limit = R * fit_fraction
    valid = np.isfinite(profile_v) & (np.abs(bin_centers) <= fit_limit)
    v0_fixed = np.nan
    if valid.sum() >= 3:
        rv = bin_centers[valid]
        vv = profile_v[valid]
        A_mat = (1.0 - (rv / R) ** 2)[:, None]
        try:
            coeffs, *_ = np.linalg.lstsq(A_mat, vv, rcond=None)
            v0_fixed = float(coeffs[0])
        except Exception:
            pass
    Q_pois_mean = v0_fixed * A_px2 / 2.0 if np.isfinite(v0_fixed) else np.nan

    return dict(
        bin_centers=bin_centers, profile_v=profile_v,
        v0_fixed=v0_fixed,
        Q_plug_mean=Q_plug_mean, Q_pois_mean=Q_pois_mean,
        Q_plug_t=None, Q_pois_t=None,
    )
