"""Waveform shape of a single recording on a common cardiac-phase axis, and shape similarity between recordings.

Each recording keeps its own heart rate f0.  Time maps to cardiac phase

    theta = f0 t + phi_1 / 2pi + peak_at   (mod 1),

so the fundamental (H1) peaks at ``peak_at`` in every recording.  Two descriptions of shape follow:

* **Folded waveform:** Q(t) averaged in phase bins.  Not limited to a few harmonics.  Folding even and odd
  cycles separately gives a split-half noise ceiling.
* **Harmonic shape coefficients** s_k = Z_k e^{-ik phi_1} / |Z_1|, k = 2, 3, where Z_k is the complex amplitude
  (Q = Re sum Z_k e^{ik w t}) and phi_1 = arg Z_1.  They are free of gain, time origin and heart rate:
  |s_2| = H2/H1 and arg s_2 = Theta_21 = phi_2 - 2 phi_1.  Uncertainty from a delete-one-cycle jackknife,
  which respects the autocorrelation of the residuals.

Similarity between recordings a and b:

* ``r``      : Pearson correlation of the folded waveforms (scale- and offset-free).
* ``r_corr`` : r / sqrt(R_a R_b), R = split-half reliability (Spearman-Brown).  About 1 when the recordings
               agree as well as their noise allows.
* ``F``      : Hotelling-type test of s^a = s^b over Re/Im of s_2, s_3 with jackknife covariances,
               small-sample corrected so F is about 1 when the shapes differ only by noise.
               Within-recording null: compare ``half_even`` with ``half_odd`` of one recording.
"""
from __future__ import annotations

import numpy as np

FPS = 250.0


def _design(t, f0, K):
    w = 2 * np.pi * f0 * t
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.column_stack(cols)


def _phasors(beta, K):
    # A cos + B sin = Re[(A - iB) e^{iwt}]
    return np.array([beta[2 * k - 1] - 1j * beta[2 * k] for k in range(1, K + 1)])


def _shape(Z):
    z1 = Z[0]
    a1 = abs(z1)
    return np.array([Z[k - 1] * np.exp(-1j * k * np.angle(z1)) / a1 for k in range(2, len(Z) + 1)])


def _fold(theta, q, n_bins):
    idx = np.minimum((theta * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    tot = np.bincount(idx, weights=q, minlength=n_bins)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _shape_fit(X, q, cyc, K):
    """Shape coefficients and their delete-one-cycle jackknife covariance (Re parts, then Im parts)."""
    beta, *_ = np.linalg.lstsq(X, q, rcond=None)
    Z = _phasors(beta, K)
    if abs(Z[0]) <= 0:
        return np.full(K - 1, np.nan + 0j), None, np.nan, 0
    s = _shape(Z)
    cycles = np.unique(cyc)
    reps, z1s = [], []
    if cycles.size >= 4:
        for c in cycles:
            m = cyc != c
            if m.sum() < X.shape[1] * 4:
                continue
            b, *_ = np.linalg.lstsq(X[m], q[m], rcond=None)
            Zj = _phasors(b, K)
            if abs(Zj[0]) > 0:
                reps.append(_shape(Zj))
                z1s.append(Zj[0])
    cov, snr1 = None, np.nan
    n = len(reps)
    if n >= 4:
        V = np.array([np.concatenate([r.real, r.imag]) for r in reps])
        cov = (n - 1) / n * ((V - V.mean(0)).T @ (V - V.mean(0)))
        z1s = np.array(z1s)
        se1 = np.sqrt((n - 1) / n * np.sum(np.abs(z1s - z1s.mean()) ** 2))
        snr1 = float(abs(Z[0]) / se1) if se1 > 0 else np.inf
    return s, cov, snr1, n


def align_recording(q, f0, n_bins=40, peak_at=0.4, K=3, fps=FPS, positive=True):
    """Shape description of one recording.  Returns None if it cannot be aligned.

    ``positive``: flip the trace so its mean is positive (display convention; shape coefficients flip sign with Q).
    """
    q = np.asarray(q, dtype=float)
    if q.size < 50 or not np.isfinite(f0) or f0 <= 0:
        return None
    t = np.arange(q.size) / fps
    ok = np.isfinite(q)
    sign = 1.0
    if positive and np.nanmean(q) < 0:
        sign = -1.0
    q = sign * q
    t, q = t[ok], q[ok]
    X = _design(t, f0, K)
    beta, *_ = np.linalg.lstsq(X, q, rcond=None)
    Z = _phasors(beta, K)
    if abs(Z[0]) <= 0:
        return None
    a0 = float(beta[0])
    phase0 = f0 * t + np.angle(Z[0]) / (2 * np.pi) + peak_at
    theta = phase0 % 1.0
    cyc = np.floor(phase0).astype(int)
    fold = _fold(theta, q, n_bins)
    even = cyc % 2 == 0
    f_even = _fold(theta[even], q[even], n_bins)
    f_odd = _fold(theta[~even], q[~even], n_bins)
    r_half = _corr(f_even, f_odd)
    R = 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else np.nan

    s, cov, snr1, ncyc = _shape_fit(X, q, cyc, K)
    halves = {}
    for name, msk in (("even", even), ("odd", ~even)):
        halves[name] = _shape_fit(X[msk], q[msk], cyc[msk], K)
    return dict(f0=float(f0), a0=a0, amp1=float(abs(Z[0])), Z=Z, s=s, s_cov=cov, snr_h1=snr1,
                fold=fold, fold_even=f_even, fold_odd=f_odd, r_half=r_half, R=R, sign=sign,
                n_cycles=float(ncyc), n_jack=ncyc, h21=float(abs(s[0])),
                half_even=dict(s=halves["even"][0], s_cov=halves["even"][1], n_jack=halves["even"][3]),
                half_odd=dict(s=halves["odd"][0], s_cov=halves["odd"][1], n_jack=halves["odd"][3]),
                th21_deg=float(np.degrees(np.angle(s[0]))))


def shape_F(a, b, ridge=1e-6):
    """Two-sample Hotelling-type test of equal shape coefficients with jackknife covariances.

    T2 = d^T (S_a + S_b)^{-1} d, p = len(d), nu = (n_a - 1) + (n_b - 1);
    F = T2 (nu - p + 1) / (p nu), approximately F(p, nu - p + 1) under equal shapes (mean ~ 1).
    Returns (F, p, nu, T2); NaNs if a covariance is missing or nu is too small.
    """
    if a.get("s_cov") is None or b.get("s_cov") is None:
        return np.nan, 0, 0, np.nan
    d = np.concatenate([(a["s"] - b["s"]).real, (a["s"] - b["s"]).imag])
    p = len(d)
    nu = (a["n_jack"] - 1) + (b["n_jack"] - 1)
    if nu - p + 1 < 2 or not np.all(np.isfinite(d)):
        return np.nan, p, nu, np.nan
    S = a["s_cov"] + b["s_cov"]
    S = S + ridge * np.trace(S) / p * np.eye(p)
    try:
        T2 = float(d @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        return np.nan, p, nu, np.nan
    return T2 * (nu - p + 1) / (p * nu), p, nu, T2


def compare(a, b, R_min=0.3):
    """Similarity of two ``align_recording`` results.

    r      : Pearson correlation of the folded waveforms;
    r_corr : r / sqrt(R_a R_b), NaN when either split-half reliability R < R_min (too noisy to correct);
    F, p_F : shape test of H2/H3 relative to H1 (see ``shape_F``) and its p-value.
    """
    from scipy.stats import f as fdist
    r = _corr(a["fold"], b["fold"])
    rc = np.nan
    if np.isfinite(r) and a["R"] >= R_min and b["R"] >= R_min:
        rc = r / np.sqrt(a["R"] * b["R"])
    F, p, nu, T2 = shape_F(a, b)
    pF = float(fdist.sf(F, p, nu - p + 1)) if np.isfinite(F) else np.nan
    dth = float(np.degrees(np.angle(a["s"][0] * np.conj(b["s"][0]))))
    return dict(r=r, r_corr=rc, F=F, p_F=pF, T2=T2, d_th21_deg=dth,
                h21_ratio=a["h21"] / b["h21"] if b["h21"] > 0 else np.nan)
