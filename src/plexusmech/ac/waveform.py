"""Log-waveform loss for the wall distensibility D (the method of the paper since 2026-11-23).

Each fitted vessel e has reconstructed harmonics Ĥ_n and a reconciled mean flow Q', and the RC network gives q_n(D) and its own
mean flow Q^s.  With h_obs,n = Ĥ_n / Q' and h_sim,n(D) = q_n(D) / Q^s (complex, each in its own mean-flow direction),

    L(D) = (1/N) Σ_e Σ_{n=1,2} [ ln(|h_sim,n| / |h_obs,n|)² + Δφ_n² ],   Δφ_n = arg(h_sim,n · conj(h_obs,n)) ∈ (−π, π]

the mean squared magnitude of the complex logarithm of h_sim / h_obs over both harmonics.  No weights, no fitted nuisance
parameters; boundary forcing as measured (no group corrections), tile clocks fixed.  One radian of phase error counts as a factor
e in amplitude.  D̂ = vertex of a parabola in log10 D through the grid minimum and its neighbours (profile.quadratic_min).
"""
from __future__ import annotations
import numpy as np
from .model import group_response
from .group_bc import groups
from .profile import D_GRID, quadratic_min


def log_waveform_distance(C, D, eta=0.0):
    """Per fitted vessel (C['elig']): squared log-waveform distance at D, shape (2 harmonics, N); columns of the sum are
    ln-amplitude² + phase² per harmonic."""
    el = C["elig"]; gid = groups(C); Qp, Qs = C["Qp"][el], C["Qs_dc"][el]; out = []
    for k in (1, 2):
        hs = group_response(C, k, D, eta, gid, el).sum(1) / Qs; ho = C["H"][k][el] / Qp       # boundary forcing uncorrected: columns summed
        out.append(np.log(np.abs(hs) / np.abs(ho)) ** 2 + np.angle(hs * np.conj(ho)) ** 2)
    return np.array(out)


def fit_log_waveform(C, d_grid=D_GRID, eta=0.0, tile=None, n_boot=1000, seed=0):
    """L(D) on d_grid and its minimum.  tile (per edge of the field): if given, D̂ is also re-fitted on resampled tiles
    (whole tiles drawn with replacement, n_boot times) — the spread of D̂ given that vessels of a tile are not independent.
    Returns dict(D_grid, L, D (grid minimum), D_parabola, n, D_boot (2.5, 50, 97.5 percentiles) or None)."""
    d_grid = np.asarray(d_grid, float); E = np.array([log_waveform_distance(C, d, eta).sum(0) for d in d_grid])     # (nD, N)
    if not np.all(np.isfinite(E)):
        raise ValueError("log-waveform loss: a fitted vessel has a vanishing observed or simulated harmonic")
    fit = lambda Lv: quadratic_min(d_grid, Lv.mean(1) - Lv.mean(1).min()); L = E.mean(1); boot = None
    if tile is not None and n_boot:
        tl = np.asarray(tile)[C["elig"]]; tiles = np.unique(tl); ix = {t: np.flatnonzero(tl == t) for t in tiles}; rng = np.random.default_rng(seed)
        bs = [fit(E[:, np.concatenate([ix[t] for t in rng.choice(tiles, len(tiles))])]) for _ in range(n_boot)]
        boot = [float(np.percentile(bs, q)) for q in (2.5, 50, 97.5)]
    return dict(D_grid=d_grid, L=L, D=float(d_grid[int(np.argmin(L))]), D_parabola=float(fit(E)), n=int(E.shape[1]), D_boot=boot)
