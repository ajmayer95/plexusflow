"""Joint (D, η) estimate with the boundary forcing corrected per group, profiled by the same likelihood.

Groups: left carve nodes, right carve nodes, V1, V2.  Each keeps its measured local-ratio pattern f0 and gets an
amplitude correction per harmonic and ONE timing shift shared by the harmonics (coherent: the boundary pulse keeps
its measured shape φ2 − 2φ1):
    s_g,k = exp(a_g,k + i k b_g)                     (12 parameters)
    H_k,sim = Σ_g s_g,k · (M_k f0_k restricted to group g)
Per vessel: phases relative to each field's own mean-flow direction, PI = |H1| / |Q̄|.
    NLL = NLL_vm(Δφ1; κ ∝ 1/c²) + NLL_vm(Δφ2; κ ∝ 1/c²) + NLL_t(log10 PI_sim/PI_data; scale ∝ c)
minimised over the 12 parameters (|s| ∈ [0.1, 4], warm-started from the neighbouring grid point) on a (D, η) grid.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .model import response
from .likelihood import nll_vm, nll_t

GROUPS = ("L carve", "R carve", "V1", "V2")
D_GRID = np.logspace(np.log10(2e-4), np.log10(1.2e-3), 14)
ETA_GRID = np.round(np.arange(0.0, 0.61, 0.1), 2)


def groups(C):
    return np.r_[np.where(C["dom"]["side"] == "L", 0, 1), [2, 3]]


def estimate(C, c, d_grid=D_GRID, eta_grid=ETA_GRID, progress=print):
    """C: model.build_context; c: per-edge reliability (array over all carved edges).
    Returns dict(S, S_fixed, best, D_grid, eta_grid, i_min, j_min, x_min)."""
    el = C["elig"]; c = c[el]; bnd = np.asarray(C["dom"]["bnd"]); gid = groups(C)
    Qp, Qs = C["Qp"][el], C["Qs_dc"][el]
    phd = {k: np.angle(C["H"][k][el] * np.sign(Qp)) for k in (1, 2)}; pid = np.abs(C["H"][1][el]) / np.abs(Qp); w = 1 / c ** 2
    sgnQs = np.sign(Qs)

    def nll_all(x, B1, B2):
        a1, a2, b = x[:4], x[4:8], x[8:]
        q1 = B1 @ np.exp(a1 + 1j * b); q2 = B2 @ np.exp(a2 + 2j * b)
        d1 = np.angle(np.exp(1j * (phd[1] - np.angle(q1 * sgnQs)))); d2 = np.angle(np.exp(1j * (phd[2] - np.angle(q2 * sgnQs))))
        r = np.log10(np.maximum(np.abs(q1) / np.maximum(np.abs(Qs), 1e-30), 1e-12) / pid)
        return nll_vm(d1, w) + nll_vm(d2, w) + nll_t(r, c)

    bounds = [(np.log(0.1), np.log(4))] * 8 + [(-np.pi, np.pi)] * 4
    nE_, nD = len(eta_grid), len(d_grid)
    S = np.full((nE_, nD), np.nan); S_fixed = np.full((nE_, nD), np.nan); best = {}; prev = None
    for j, D_ in enumerate(d_grid):
        for i, eta in enumerate(eta_grid):
            Bs = {}
            for k in (1, 2):
                f0 = C["F"][(k, "flux")][bnd]; M = response(C, k, D_, eta, bnd)[el]
                Bs[k] = np.column_stack([M[:, gid == g] @ f0[gid == g] for g in range(4)])
            S_fixed[i, j] = nll_all(np.zeros(12), Bs[1], Bs[2])
            starts = [np.zeros(12)] + ([prev] if prev is not None else [])
            rb = min((minimize(nll_all, x0, args=(Bs[1], Bs[2]), method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300))
                      for x0 in starts), key=lambda r: r.fun)
            prev = rb.x; S[i, j] = rb.fun; best[(i, j)] = rb.x
        if progress:
            progress(f"  D {D_:.2e} done (best so far {np.nanmin(S):.0f})")
    i, j = np.unravel_index(np.nanargmin(S), S.shape)
    return dict(S=S, S_fixed=S_fixed, best=best, D_grid=np.asarray(d_grid), eta_grid=np.asarray(eta_grid), i_min=i, j_min=j, x_min=best[(i, j)])


def group_summary(C, fb, x):
    """Per group: timing shift (beats), H1/H2 amplitude corrections and the implied pulsatility |f_k| / |f_DC|."""
    bnd = np.asarray(C["dom"]["bnd"]); gid = groups(C); out = {}
    for n, g in enumerate(GROUPS):
        row = dict(timing=x[8 + n] / (2 * np.pi))
        for k in (1, 2):
            a = x[(k - 1) * 4 + n]; f0 = C["F"][(k, "flux")][bnd]
            row[f"amp{k}"] = float(np.exp(a)); row[f"PI{k}"] = float(abs(np.exp(a) * f0[gid == n].sum()) / abs(fb[bnd][gid == n].sum()))
        out[g] = row
    return out
