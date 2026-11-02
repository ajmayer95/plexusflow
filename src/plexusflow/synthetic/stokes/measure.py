"""Run the PRODUCTION per-vessel measurement (optical flow → Kloosterman Poiseuille flux) on a synthetic video."""
from __future__ import annotations

import numpy as np

from .render import FDT


def run_measurement(vid, cl, R_seg, f0, mask, fdt=FDT, **of_overrides):
    """compute_of_profile_and_Q with the batch Farnebäck settings (+ overrides such as dt / dt_max / target_disp_px)."""
    from ...analyze.profile import compute_of_profile_and_Q
    from ...analyze.mosaic_piv import FB_KWARGS
    kw = dict(FB_KWARGS); kw.update(of_overrides)
    return compute_of_profile_and_Q(vid, cl, float(R_seg), f0_hz=f0, frame_dt_s=fdt, ridge_mask=mask, **kw)


def flux_record(u, R_used_default):
    """Mean flux, Kloosterman kept fraction, slices, radius and the batch quality numbers from one measurement."""
    from ...analyze.mosaic_piv import _flow_quality
    q = float(u.get("Q_kloosterman", np.nan))
    psq = np.asarray(u.get("Q_kloosterman_per_slice", []), float)
    return dict(Q_kl=q, f=float(u.get("Q_kloosterman_valid_frac", 0.0)) if np.isfinite(q) else 0.0, n_slices=int(psq.size),
                R_kl=float(u.get("R_kloosterman", np.nan)), Q_t=np.asarray(u.get("Q_pois_t", []), np.float32),
                **_flow_quality(u.get("vfield_mean"), u.get("cl_used"), float(u.get("R_used", R_used_default))))


def h1_over_dc(u, f0, dt, fdt=FDT):
    """Measured H1/DC (complex) from Q_pois_t, referenced to the true waveform's time origin.  Frame f shows tracers at
    frame-time f + 0.5, so pair (t, t + dt) measures the flux over [t + 0.5, t + dt + 0.5], centred at t + 0.5 + dt/2."""
    from ...analyze.harmonic import fit_harmonics
    qt = np.asarray(u.get("Q_pois_t", []), float)
    used = int(u.get("of_dt", dt if isinstance(dt, int) else -1))
    if qt.size <= 50 or np.isfinite(qt).sum() <= 50:
        return np.nan + 0j, used
    hr = fit_harmonics(qt, fdt, f0, K=2, loss="huber", include_dc=True)
    H = next(h for h in hr["harmonics"] if h["k"] == 1)
    if not hr["a0"]:
        return np.nan + 0j, used
    return (H["A"] - 1j * H["B"]) / hr["a0"] * np.exp(-1j * 2 * np.pi * f0 * fdt * (0.5 + used / 2)), used


def central_dip(vf, cl, R):
    """Smallest per-slice ratio of axial speed at the centreline to the slice's peak speed (|ρ| ≤ R), slices smoothed
    over ±2: ≈ 1 for an uninterrupted vessel, → 0 where an obstacle splits the flow."""
    from ...analyze.centerline import _centerline_geometry
    if vf is None or cl is None:
        return np.nan
    vf = np.asarray(vf, float); cl = np.asarray(cl, float); _, tang, norm = _centerline_geometry(cl); rho = np.linspace(-R, R, 21); r = []
    H, W = vf.shape[:2]
    for i in range(len(cl)):
        rr = np.clip(np.round(cl[i, 0] + rho * norm[i, 0]).astype(int), 0, H - 1); cc = np.clip(np.round(cl[i, 1] + rho * norm[i, 1]).astype(int), 0, W - 1)
        ax = np.abs(vf[rr, cc, 1] * tang[i, 0] + vf[rr, cc, 0] * tang[i, 1]); r.append((ax[8:13].mean(), ax.max()))
    r = np.array(r); k = np.ones(5) / 5; c = np.convolve(r[:, 0], k, "valid"); m = np.convolve(r[:, 1], k, "valid")
    return float(np.min(c / np.maximum(m, 1e-12)))
