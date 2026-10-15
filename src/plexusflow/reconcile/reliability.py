"""Reliability scale of a Kloosterman-valid measurement, calibrated on same-tile junction balance.

    c_e = c_T(tile) + c_f (1 − f_e) + c_a (1 − align_e),      σ_e = √((|Q_e| c_e)² + σ0²)

f      fraction of Kloosterman cross-sections kept (quality.f)
align  speed-weighted |cos θ| between the time-mean OF vector and the centreline tangent within ±R_seg
       (quality.align; missing → 1)
Fitted by maximum likelihood on junctions where every incident edge has a Kloosterman-valid measurement from the
same tile: net flux Σ s·Q ~ N(0, Σ Q² c² + σ0²).  c is an empirical reliability scale, not a standard error.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def measurement_table(G):
    """{(tile, (u, v)): row} for every Kloosterman-valid measurement; Q is the time-mean flux signed u→v."""
    M = {}
    for u, v, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            q = m.get("quality") or {}
            if not q.get("kl_ok"):
                continue
            mq = (m.get("harm") or {}).get("mean_Q")
            if mq is None or not np.isfinite(mq):
                continue
            Q = mq if m["flow_from"] == u else -mq
            a = q.get("align")
            M[(int(m["tile_id"]), (u, v))] = dict(tile=int(m["tile_id"]), Q=float(Q), f=float(q["f"]),
                                                   align=float(a) if a is not None and np.isfinite(a) else np.nan,
                                                   R_kl_um=m.get("R_kl_um"))
    return M


def same_tile_junctions(G, tile, measured):
    """Junctions of one tile: lists of (edge, +1 if the edge leaves the node else −1) at non-boundary nodes of
    degree ≥ 3 whose incident edges are ALL in ``measured`` (a set of (u, v))."""
    J = []
    for n in {n for e in measured for n in e}:
        if G.nodes[n].get("boundary_type") in ("source", "sink") or G.degree(n) < 3:
            continue
        inc = []
        for nb in G.neighbors(n):
            e = (n, nb) if (n, nb) in measured else ((nb, n) if (nb, n) in measured else None)
            if e is None:
                break
            inc.append((e, 1.0 if e[0] == n else -1.0))
        else:
            J.append(inc)
    return J


def _align(m):
    return m["align"] if np.isfinite(m["align"]) else 1.0


def _arrays(J, M, tiles):
    idx, dc, f, al, ti, net = [], [], [], [], [], []
    for k, (t, j) in enumerate(J):
        tot = 0.0
        for e, s in j:
            m = M[(t, e)]; idx.append(k); dc.append(abs(m["Q"])); f.append(m["f"]); al.append(_align(m))
            ti.append(tiles.index(t)); tot += s * m["Q"]
        net.append(tot)
    return tuple(np.array(x) for x in (idx, dc, f, al, ti, net))


def _c(th, f, al, ti, nT):
    return np.exp(th[ti]) + np.exp(th[nT]) * (1 - f) + np.exp(th[nT + 1]) * (1 - al)


def _nll(th, d, nT, tau=1.0, grad=True):
    """Negative log-likelihood of junction imbalances (+ a mild Gaussian pull of the per-tile log c_T toward their mean,
    SD `tau`), and its exact gradient.  θ = (log c_T[0..nT-1], log c_f, log c_a, log σ0)."""
    idx, dc, f, al, ti, net = d
    cT, cf, ca, s0 = np.exp(th[:nT]), np.exp(th[nT]), np.exp(th[nT + 1]), np.exp(th[-1])
    c = cT[ti] + cf * (1 - f) + ca * (1 - al)
    V = np.bincount(idx, dc ** 2 * c ** 2, minlength=len(net)) + s0 ** 2
    a = th[:nT]; dev = a - a.mean()
    val = 0.5 * np.sum(np.log(2 * np.pi * V) + net ** 2 / V) + 0.5 * np.sum(dev ** 2) / tau ** 2
    if not grad:
        return val
    r = 0.5 * (1 / V - net ** 2 / V ** 2)                    # dNLL/dV_k
    q = r[idx] * 2 * dc ** 2 * c                              # dNLL/dc_i
    g = np.empty_like(th)
    g[:nT] = np.bincount(ti, q, minlength=nT) * cT + dev / tau ** 2
    g[nT] = np.sum(q * (1 - f)) * cf
    g[nT + 1] = np.sum(q * (1 - al)) * ca
    g[-1] = np.sum(r) * 2 * s0 ** 2
    return val, g


def fit_reliability(G, M, tau=1.0):
    """ML fit of (c_T per tile, c_f, c_a, σ0) on all same-tile junctions.  Returns (model dict, junction list).

    Well-posed by construction (2026-10-15; the earlier fit used numerical gradients and default tolerances, and its
    weakly constrained parameters — σ0, c_T of tiles with few junctions — differed by up to 15–160 % between scipy
    versions, which moved the sensitive ~1 % of reconciled flows): exact gradient, tight tolerances, σ0 ≥ 1e-4·median|Q|,
    c in [1e-4, 10], and a mild pull of log c_T toward the tiles' mean (SD `tau` in log, i.e. a factor ~e) so tiles with
    few junctions do not wander."""
    tiles = sorted({t for t, _ in M}); nT = len(tiles); J = []
    for t in tiles:
        J += [(t, j) for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t})]
    d = _arrays(J, M, tiles)
    qmed = float(np.median(d[1])) if len(d[1]) else 1.0
    bounds = [(np.log(1e-4), np.log(10.0))] * (nT + 2) + [(np.log(1e-4 * qmed), np.log(10.0 * qmed))]
    x0 = np.r_[np.full(nT, np.log(0.1)), np.log(0.5), np.log(0.1), np.log(max(5e-3, 2e-4 * qmed))]
    best = None
    for jit in (0.0, 0.3, -0.3):
        x = np.clip(x0 + jit, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(_nll, x, args=(d, nT, tau), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(ftol=1e-15, gtol=1e-9, maxiter=100000, maxcor=50))
        best = r if best is None or r.fun < best.fun else best
    th = best.x
    model = dict(c_T={int(t): float(np.exp(th[i])) for i, t in enumerate(tiles)}, c_f=float(np.exp(th[nT])),
                 c_a=float(np.exp(th[nT + 1])), sigma0=float(np.exp(th[-1])), n_junctions=len(J),
                 tau_log_cT=tau, converged=bool(best.success), nll=float(best.fun),
                 formula="c = c_T + c_f*(1-f) + c_a*(1-align);  sigma = sqrt((|Q| c)^2 + sigma0^2)")
    return model, J


def c_of(model, m):
    """Reliability c of one measurement row (needs keys tile, f, align)."""
    return model["c_T"][m["tile"]] + model["c_f"] * (1 - m["f"]) + model["c_a"] * (1 - _align(m))


def c_of_measurement(model, m):
    """Reliability c of a raw graph measurement dict (``measurements_piv`` entry with a ``quality`` record)."""
    q = m["quality"]
    return c_of(model, dict(tile=int(m["tile_id"]), f=q["f"], align=q["align"] if np.isfinite(q["align"]) else np.nan))
