"""Motion diagnostics of a recording, independent of the flux estimator (2026-11-17).

Three questions are kept apart:
  1. is blood motion detectable?            — no velocity-profile assumption
  2. what is its temporal character?        — no flux needed
  3. can it be converted to volumetric flux? — profile-fit profile fit (analyze.profile_flux), a separate question

Signal: the vessel-aligned velocity series v(t) — a fixed spatial average of the streamwise optical flow over the vessel
pixels (profile.py's pooled fixed-radius series, in any units).  Every quantity below is invariant to a multiplicative
calibration of v.  Model at the tile heart rate f0 (from the batch, independent of this recording's fit):

    v(t) = v0 + Σ_{n=1..K} Re[v̂_n e^{inωt}] + ε(t),   K = 3

  z_dc      |v0| / SE(v0); SE from the long-run (HAC, Bartlett kernel, bandwidth one beat) variance of ε — accounts for
            temporal autocorrelation and low-frequency noise.  Detectable directed motion (operational: z_dc ≥ 3).
  chi_ac    Σ_n |v̂_n|² / Var(v̂_n), 2K degrees of freedom; Var from the residual periodogram within ±0.3 f0 of n f0 (the
            harmonic bins excluded).  p_ac its χ² tail probability, z_ac = √(chi_ac / 2K).  Detectable coherent cardiac
            motion (operational: p_ac < 1e-3).
  eta_osc   ½Σ|v̂_n|² / (v0² + ½Σ|v̂_n|²) ∈ [0, 1]: fraction of the fitted mean-square velocity that is cardiac
            (0 steady, 0.5 equal, 1 purely oscillatory); bounded as the mean → 0, unlike |H1|/|mean|.  Interpret only
            where motion is detectable.
  reversal  the fitted waveform exceeds +3 sd and −3 sd within a beat (sd = pointwise SE of the fit): bidirectional motion.

Calibration caveats (21S checks, sandbox/motion_*.py): with 3–9 beats per recording and structured optical-flow noise the
p-values are NOT calibrated (real-data off-frequency nulls 0.5–2.6 % at nominal 0.1 %; synthetic stationary / slow 0 %,
fast steady 8 %), so the thresholds are operational detection criteria, not formal significance.  Optical flow gives
reversing motion a small spurious mean (synthetic zero-mean oscillation: z_dc ≥ 3 in 11 of 12), so detectable DC is a
property of the measurement, not proof of physical net flow.  Diagnostics only — no acceptance rule uses them.
"""
from __future__ import annotations

import numpy as np

Z_DC_DETECT = 3.0
P_AC_DETECT = 1e-3


def motion_stats(v, f0_hz, fs_hz, K=3, z_rev=3.0, f_test=None):
    """Motion diagnostics of the series ``v`` sampled at ``fs_hz`` (see module doc); None if < 100 finite samples or no
    heart rate.  ``f_test`` replaces f0 in the harmonic model (null tests at a non-cardiac frequency)."""
    from scipy import stats
    v = np.asarray(v, float); v = v[np.isfinite(v)]; T = len(v)
    if T < 100 or f0_hz is None or not np.isfinite(f0_hz) or f0_hz <= 0:
        return None
    f = float(f0_hz if f_test is None else f_test)
    t = np.arange(T) / fs_hz; w = 2 * np.pi * f
    X = np.column_stack([np.ones(T)] + [g(k * w * t) for k in range(1, K + 1) for g in (np.cos, np.sin)])
    b, *_ = np.linalg.lstsq(X, v, rcond=None); e = v - X @ b
    L = max(int(round(fs_hz / f0_hz)), 1)
    gam = [float(np.dot(e[:T - l], e[l:]) / T) for l in range(L + 1)]
    lrv = gam[0] + 2 * sum((1 - l / (L + 1)) * gam[l] for l in range(1, L + 1))
    se0 = np.sqrt(max(lrv, gam[0] * 1e-3) / T)
    fr = np.fft.rfftfreq(T, 1 / fs_hz); Pw = np.abs(np.fft.rfft(e)) ** 2 / T
    chi = ms_ac = var_sum = 0.0
    for k in range(1, K + 1):
        band = (np.abs(fr - k * f) <= 0.3 * f) & (np.abs(fr - k * f) > 1.5 * fs_hz / T)
        s2 = float(np.mean(Pw[band])) if band.sum() >= 3 else gam[0]
        vc = max(2 * s2 / T, 1e-300); A, B = b[2 * k - 1], b[2 * k]
        chi += (A ** 2 + B ** 2) / vc; ms_ac += (A ** 2 + B ** 2) / 2; var_sum += vc
    sd = np.sqrt(se0 ** 2 + var_sum)
    tt = np.linspace(0, 1 / f, 200)
    vf = b[0] + sum(b[2 * k - 1] * np.cos(k * w * tt) + b[2 * k] * np.sin(k * w * tt) for k in range(1, K + 1))
    den = b[0] ** 2 + ms_ac
    return dict(z_dc=float(abs(b[0]) / se0) if se0 > 0 else float("nan"), chi_ac=float(chi),
                p_ac=float(stats.chi2.sf(chi, 2 * K)), z_ac=float(np.sqrt(chi / (2 * K))),
                eta_osc=float(ms_ac / den) if den > 0 else float("nan"),
                reversal=bool(vf.max() - z_rev * sd > 0 and vf.min() + z_rev * sd < 0),
                n_beats=float(T * f0_hz / fs_hz), f0_hz=float(f0_hz))


def regime(ms, z_dc=Z_DC_DETECT, p_ac=P_AC_DETECT):
    """'directed+cardiac' | 'steady' | 'oscillatory' | 'unresolved' (operational thresholds; None → 'unresolved')."""
    if not ms:
        return "unresolved"
    d, a = ms["z_dc"] >= z_dc, ms["p_ac"] < p_ac
    return "directed+cardiac" if d and a else "steady" if d else "oscillatory" if a else "unresolved"
