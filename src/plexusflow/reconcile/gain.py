"""Recording-scale (heart-rate) correction of mean fluxes (adopted 2026-11-18).

Recordings of one vessel made at different times differ systematically in flux: across the 21S acquisition the flux of a
vessel scales with the recording's heart rate as ≈ f^2.4 (repeat recordings of the same vessels; significant, cross-validated
on held-out vessels, the same temporal pattern in independent vessel subsets).  The mean-flow input is therefore normalised to
a common reference heart rate:

    y_ref = y · (f_ref / f_t)^β,      log|y_kt| = α_k + β log(f_t / f_ref) + ε    (α_k one per vessel)

β is fitted by robust (Huber-IRLS) least squares on all Kloosterman-valid recordings of vessels recorded in ≥ 2 tiles; f_ref
= median tile heart rate.  β is an EMPIRICAL scaling, not a physiological law.  Relative errors c are scale-free, and the
absolute terms of the reconstruction (σ0, between-recording spread) are computed from the corrected fluxes, so the
uncertainties transform with the data.  Pulsatile analysis is unaffected: it uses each recording's waveform relative to its own
mean, and within the analysed heart-rate band a vessel's pulsatility fraction does not depend on heart rate or flux scale.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT = "hr"                    # "hr" | None


def tile_heart_rates(G):
    """{tile: f0 (Hz)} from the measurements' harmonic fits (one heart rate per recording)."""
    f = {}
    for *_, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            v = (m.get("harm") or {}).get("f0_hz")
            if v is not None and np.isfinite(v) and v > 0:
                f.setdefault(int(m["tile_id"]), float(v))
    return f


def fit_hr_gain(M, f0, n_iter=10):
    """β and f_ref of y ∝ f^β from repeat recordings in the measurement table M ({(tile, edge): row with Q}).
    Returns dict(beta, f_ref, n_vessels, n_recordings); beta = 0 when there is no heart-rate spread or no repeats."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    by = defaultdict(dict)
    for (t, e), m in M.items():
        if m["Q"] != 0 and t in f0:
            by[e][t] = abs(m["Q"])
    V = [e for e, r in by.items() if len(r) >= 2]
    fref = float(np.median(list(f0.values()))) if f0 else np.nan
    if not V:
        return dict(beta=0.0, f_ref=fref, n_vessels=0, n_recordings=0)
    rv, x, y = [], [], []
    for k, e in enumerate(V):
        for t, q in by[e].items():
            rv.append(k); x.append(np.log(f0[t] / fref)); y.append(np.log(q))
    rv, x, y = np.array(rv), np.array(x), np.array(y); N = len(y)
    if np.ptp(x) < 1e-9:
        return dict(beta=0.0, f_ref=fref, n_vessels=len(V), n_recordings=N)
    X = sp.hstack([sp.csr_matrix((np.ones(N), (np.arange(N), rv)), shape=(N, len(V))), sp.csr_matrix(x[:, None])]).tocsr()
    w = np.ones(N)
    for _ in range(n_iter):
        W = sp.diags(np.sqrt(w)); b = spla.lsqr(W @ X, np.sqrt(w) * y, atol=1e-12, btol=1e-12, iter_lim=50000)[0]
        r = y - X @ b; s = 1.4826 * np.median(np.abs(r)); w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12))
    return dict(beta=float(b[-1]), f_ref=fref, n_vessels=len(V), n_recordings=N)


def apply_hr_gain(M, f0, fit):
    """Copy of M with every flux normalised to the reference heart rate (rows without a heart rate are left unchanged)."""
    b, fr = fit["beta"], fit["f_ref"]
    return {k: (dict(m, Q=m["Q"] * (fr / f0[k[0]]) ** b, gain=(f0[k[0]] / fr) ** b) if k[0] in f0 else m) for k, m in M.items()}
