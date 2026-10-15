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


def _nll(th, d, nT):
    idx, dc, f, al, ti, net = d
    V = np.bincount(idx, dc ** 2 * _c(th, f, al, ti, nT) ** 2, minlength=len(net)) + np.exp(2 * th[-1])
    return 0.5 * np.sum(np.log(2 * np.pi * V) + net ** 2 / V)


def fit_reliability(G, M):
    """ML fit of (c_T per tile, c_f, c_a, σ0) on all same-tile junctions.  Returns (model dict, junction list)."""
    tiles = sorted({t for t, _ in M}); nT = len(tiles); J = []
    for t in tiles:
        J += [(t, j) for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t})]
    d = _arrays(J, M, tiles)
    x0 = np.r_[np.full(nT, np.log(0.1)), np.log(0.5), np.log(0.1), np.log(5e-3)]
    best = None
    for jit in (0.0, 0.3, -0.3):
        r = minimize(_nll, x0 + jit, args=(d, nT), method="L-BFGS-B")
        best = r if best is None or r.fun < best.fun else best
    th = best.x
    model = dict(c_T={int(t): float(np.exp(th[i])) for i, t in enumerate(tiles)}, c_f=float(np.exp(th[nT])),
                 c_a=float(np.exp(th[nT + 1])), sigma0=float(np.exp(th[-1])), n_junctions=len(J),
                 formula="c = c_T + c_f*(1-f) + c_a*(1-align);  sigma = sqrt((|Q| c)^2 + sigma0^2)")
    return model, J


def c_of(model, m):
    """Reliability c of one measurement row (needs keys tile, f, align)."""
    return model["c_T"][m["tile"]] + model["c_f"] * (1 - m["f"]) + model["c_a"] * (1 - _align(m))


def c_of_measurement(model, m):
    """Reliability c of a raw graph measurement dict (``measurements_piv`` entry with a ``quality`` record)."""
    q = m["quality"]
    return c_of(model, dict(tile=int(m["tile_id"]), f=q["f"], align=q["align"] if np.isfinite(q["align"]) else np.nan))
