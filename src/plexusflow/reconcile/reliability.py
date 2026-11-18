"""Reliability scale of a Kloosterman-valid measurement, calibrated on same-tile junction balance.

    c_e = c0 + c_a (1 − align_e),      σ_e = √((|Q_e| c_e)² + σ0²)          (production since 2026-11-18, PROFILE_TERM "none")
    c_e = c0 + c_p (1 − p_e) + c_a (1 − align_e)                            (PROFILE_TERM "p", 2026-11-10 … 11-17; sensitivity)

Alignment alone reconstructs as well as alignment + parabolicity (21S leave-one-recording-out 1.426× vs 1.415×, 10 %
withholding 1.409× vs 1.414×, 25 % 1.468× vs 1.475×) with one feature fewer; a constant c is clearly worse (1.491×, 1.507×,
1.540×) and miscalibrated (RMS z 0.47 / 1.52 in the best / worst fifth of junctions).  sandbox/quality_models.py.

p      parabolicity: mean over all Kloosterman cross-sections of clip(R², 0, 1), unfitted sections 0 (quality.p;
       analyze.kloosterman.parabolicity).  Profile term since 2026-11-10 (PROFILE_TERM); before, f = fraction of
       cross-sections kept (quality.f).  p and f rank recordings almost identically (Spearman 0.97 / 0.95) but p predicts
       held-out same-tile junction imbalance better (5-fold tile CV ΔNLL 21S −21.9, 27S −24.0, same number of parameters);
       f adds nothing once p is in (sandbox/parabolicity.py).  Old model files (no "profile_term") use f.
align  speed-weighted |cos θ| between the time-mean OF vector and the centreline tangent within ±R_seg
       (quality.align; missing → 1)
Fitted by maximum likelihood on junctions where every incident edge has a Kloosterman-valid measurement from the
same tile: net flux Σ s·Q ~ N(0, Σ Q² c² + σ0²).  c is an empirical reliability scale, not a standard error.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

PROFILE_TERM = "none"                      # "none" (alignment only) | "p" | "f"


def _term(model):
    return model.get("profile_term", "f")


def measurement_table(G, gain="default"):
    """{(tile, (u, v)): row} for every Kloosterman-valid measurement; Q is the time-mean flux signed u→v.
    gain: "default" → reconcile.gain.DEFAULT ("hr": fluxes normalised to the reference heart rate, the fit stored in
    LAST_GAIN[id(G)]); None → raw fluxes."""
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
            M[(int(m["tile_id"]), (u, v))] = dict(tile=int(m["tile_id"]), Q=float(Q), f=float(q["f"]), p=float(q["p"]) if np.isfinite(q.get("p", np.nan)) else float(q["f"]),   # p missing (no cached inputs): f
                                                   align=float(a) if a is not None and np.isfinite(a) else np.nan,
                                                   R_kl_um=m.get("R_kl_um"), none=1.0)
    from . import gain as _g
    kind = _g.DEFAULT if gain == "default" else gain
    if kind == "hr":
        f0 = _g.tile_heart_rates(G); fit = _g.fit_hr_gain(M, f0); LAST_GAIN[id(G)] = fit
        M = _g.apply_hr_gain(M, f0, fit)
    elif kind is not None:
        raise ValueError(f"unknown gain model {kind!r}")
    return M


LAST_GAIN = {}                    # id(G) → heart-rate gain fit of the last measurement_table(G) call


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


def _arrays(J, M, tiles, term=PROFILE_TERM):
    # term "none": the profile column is identically 1, so its coefficient drops out (alignment-only model)
    idx, dc, f, al, ti, net = [], [], [], [], [], []
    for k, (t, j) in enumerate(J):
        tot = 0.0
        for e, s in j:
            m = M[(t, e)]; idx.append(k); dc.append(abs(m["Q"])); f.append(m.get(term, 1.0)); al.append(_align(m))
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


def _fit_theta(d, nT, tau=1.0, jitters=(0.0, 0.3, -0.3)):
    """Bounded L-BFGS-B with exact gradient from a few starts; returns the best scipy result."""
    qmed = float(np.median(d[1])) if len(d[1]) else 1.0
    bounds = [(np.log(1e-4), np.log(10.0))] * (nT + 2) + [(np.log(1e-4 * qmed), np.log(10.0 * qmed))]
    x0 = np.r_[np.full(nT, np.log(0.1)), np.log(0.5), np.log(0.1), np.log(max(5e-3, 2e-4 * qmed))]
    best = None
    for jit in jitters:
        x = np.clip(x0 + jit, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(_nll, x, args=(d, nT, tau), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(ftol=1e-15, gtol=1e-9, maxiter=100000, maxcor=50))
        best = r if best is None or r.fun < best.fun else best
    return best


def fit_reliability(G, M, tau=1.0, tile_terms=False, term=PROFILE_TERM):
    """ML fit of the reliability model on all same-tile junctions.  Returns (model dict, junction list).

    Default (2026-10-19): c = c0 + c_f (1 − f) + c_a (1 − align) — three numbers.  Against the earlier per-tile model
    (tile_terms=True: c_T per tile, shrunk to their mean with SD `tau` in log) it loses 62 NLL on 21S's 4358 junctions for
    52 fewer parameters, stays calibrated across deciles, and gives the same reconciliation (held-out error 0.709 vs 0.708,
    2.7 % directions reversed both).  A residual off-centring term (quality.centre) was tested and added nothing in 5-fold
    cross-validation (held-out ΔNLL −2.8).  Fit: exact gradient, tight tolerances, σ0 ≥ 1e-4·median|Q|, c in [1e-4, 10]."""
    tiles = sorted({t for t, _ in M}); J = []
    for t in tiles:
        J += [(t, j) for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t})]
    d = _arrays(J, M, tiles, term)
    if not tile_terms:
        d = d[:4] + (np.zeros_like(d[4]),) + d[5:]
    nT = len(tiles) if tile_terms else 1
    best = _fit_theta(d, nT, tau)
    th = best.x
    cT = {int(t): float(np.exp(th[i if tile_terms else 0])) for i, t in enumerate(tiles)}
    model = dict(c_T=cT, profile_term=term, **{f"c_{term}": (0.0 if term == "none" else float(np.exp(th[nT])))}, c_a=float(np.exp(th[nT + 1])), sigma0=float(np.exp(th[-1])), n_junctions=len(J),
                 tile_terms=bool(tile_terms), converged=bool(best.success), nll=float(best.fun),
                 formula=(("c = c0 + c_a*(1-align)" if term == "none" else
                           f"c = c_T + c_{term}*(1-{term}) + c_a*(1-align)" if tile_terms else f"c = c0 + c_{term}*(1-{term}) + c_a*(1-align)"))
                         + ";  sigma = sqrt((|Q| c)^2 + sigma0^2)")
    if tile_terms:
        model["tau_log_cT"] = tau
    else:
        model["c0"] = float(np.exp(th[0]))
    return model, J


def c_of(model, m):
    """Reliability c of one measurement row (needs keys tile, align and the model's profile term, p or f)."""
    base = model["c_T"].get(m["tile"], model.get("c0")) if "c0" in model else model["c_T"][m["tile"]]
    t = _term(model)
    prof = 0.0 if t == "none" else model[f"c_{t}"] * (1 - m[t])
    return base + prof + model["c_a"] * (1 - _align(m))


def profile_coef(model):
    """(name, coefficient) of the model's profile term: ("p", c_p) or ("f", c_f)."""
    t = _term(model); return t, model[f"c_{t}"]


def c_of_measurement(model, m):
    """Reliability c of a raw graph measurement dict (``measurements_piv`` entry with a ``quality`` record)."""
    q = m["quality"]
    pv = q.get("p", np.nan); pv = pv if pv is not None and np.isfinite(pv) else q["f"]          # p missing → f (as measurement_table)
    return c_of(model, dict(tile=int(m["tile_id"]), f=q["f"], p=pv, none=1.0, align=q["align"] if np.isfinite(q["align"]) else np.nan))
