"""Mean-flow reconstruction under different measurement-uncertainty models, scored identically (paper SI).

Every variant reconstructs the same carved network from the same per-recording Kloosterman fluxes; only the weights differ.
  average          no conservation: each vessel's recording average (the measurements as they are)
  uniform          conservative projection, every measured vessel equal weight
  fixed            projection with σ = c|Q̄| ⊕ σ0 fixed from the MEASURED value (c = junction-calibrated constant)
  repeat           projection with σ from the agreement of repeated recordings, pooled by |Q̄|: s² = a²Q̄² + b²
  constant         quasi-likelihood, σ = c|Q′| ⊕ σ0 evaluated at the reconstructed flux (one number)
  cv  (primary)    quasi-likelihood, c = c0 + c_cv·CV, CV = residual temporal SD / |mean| of the recording's Q(t)
  alignment        quasi-likelihood, c = c0 + c_a(1 − alignment)
  abs+rel          quasi-likelihood, σ² = α²Q′² + β²s_k² (s_k = residual temporal SD, nL/s)
Scores (all held-out recordings, no model-specific "trusted" subset): leave-one-recording-out (5 folds, dc.repeat_folds;
all parameters refitted per fold), 10 % and 25 % of measured vessels withheld (3 draws each): median fold error
|log(Q′/Q_held)| as a factor, direction agreement.  Also total inflow and left share on the full data.
Writes results/dc_uncertainty_models.json.  Read-only on the graph.
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plexusflow.paths as P
from plexusflow.reconcile import measurement_table, carved_domain, reconcile_dc, fit_reliability
from plexusflow.reconcile.reliability import same_tile_junctions
from plexusflow.reconcile.network import emergent, project, EPS_UNIMAGED
from plexusflow.reconcile.dc import repeat_folds, ql_solve, vessel_input
from dc_withholding_curve import fit_constant

T0 = time.time()
log = lambda s: print(f"[+{time.time() - T0:5.0f} s] {s}", flush=True)
G = P.load_graph(); M = measurement_table(G, gain=None); dom = carved_domain(G)
edges = [tuple(e) for e in dom["edges"]]; ix = {e: i for i, e in enumerate(edges)}; me = len(edges)
inj, sd = np.asarray(dom["inj"]), np.asarray(dom["side"])
FS = 250.0


def resid_sd(m_graph):
    q = m_graph.get("Q_t"); f0 = (m_graph.get("harm") or {}).get("f0_hz")
    if q is None or not f0:
        return np.nan
    q = np.asarray(q, float); t = np.arange(len(q)) / FS; w = 2 * np.pi * f0
    X = np.column_stack([np.ones(len(q))] + [g(k * w * t) for k in (1, 2, 3) for g in (np.cos, np.sin)])
    return float(np.std(q - X @ np.linalg.lstsq(X, q, rcond=None)[0]))


S_ABS = {}
for u, v, d in G.edges(data=True):
    for m in d.get("measurements_piv") or []:
        k = (int(m["tile_id"]), (u, v))
        if k in M:
            S_ABS[k] = resid_sd(m)
smed = np.nanmedian(list(S_ABS.values()))
for k, m in M.items():
    m["s_abs"] = S_ABS.get(k, smed) if np.isfinite(S_ABS.get(k, np.nan)) else smed


def averaged(Mx, c_fun):
    """Per-vessel input with recording weights 1/c² (dc.vessel_input), plus imaged mask and between-recording spread."""
    recs = defaultdict(list)
    for (t, e), m in Mx.items():
        recs[e].append((m, c_fun(m)))
    y = np.zeros(me); C = np.ones(me); sb = np.zeros(me); im = np.zeros(me, bool); n = np.zeros(me, int)
    for e, r in recs.items():
        i = ix.get(e)
        if i is None:
            continue
        y[i], C[i], sb[i] = vessel_input(r); im[i] = True; n[i] = len(r)
    return y, C, sb, im, n


def proj(y, w, im):
    return project(np.where(im, y, 0.0), np.where(im, w, EPS_UNIMAGED * np.median(w[im])), dom)


def junction_constant(Mx):
    return fit_constant(G, Mx, fit_reliability(G, Mx, term="none")[0])[0]


def pooled_repeat(Mx):
    by = defaultdict(list)
    for (t, e), m in Mx.items():
        by[e].append(m["Q"])
    rows = np.array([(abs(np.mean(q)), np.var(q, ddof=1)) for q in by.values() if len(q) >= 2])
    qb = np.quantile(rows[:, 0], np.linspace(0, 1, 11)); b = np.clip(np.digitize(rows[:, 0], qb[1:-1]), 0, 9)
    xm = np.array([np.mean(rows[b == k, 0] ** 2) for k in range(10)]); ym = np.array([np.mean(rows[b == k, 1]) for k in range(10)])
    a2, b2 = np.maximum(np.linalg.lstsq(np.column_stack([xm, np.ones(10)]), ym, rcond=None)[0], 1e-8)
    return a2, b2


def fit_absrel(Mx):
    """α, β, σ0 by ML on same-tile junctions: Var(net) = Σ α²Q² + β²s² + σ0²."""
    tiles = sorted({t for t, _ in Mx}); J = []
    for t in tiles:
        for j in same_tile_junctions(G, t, {e for (tt, e) in Mx if tt == t}):
            J.append((np.array([s * Mx[(t, e)]["Q"] for e, s in j]), np.array([Mx[(t, e)]["s_abs"] for e, s in j])))
    def nll(th):
        a, b, s0 = np.exp(th)
        return sum(0.5 * (np.log(V) + y.sum() ** 2 / V) for y, s in J for V in [np.sum(a ** 2 * y ** 2 + b ** 2 * s ** 2) + s0 ** 2])
    return np.exp(minimize(nll, np.log([0.3, 0.2, 1e-3]), method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-4)).x)


def solve(Mx, kind):
    if kind == "average":
        y, C, sb, im, n = averaged(Mx, lambda m: 1.0); return np.where(im, y, np.nan)
    if kind == "uniform":
        y, C, sb, im, n = averaged(Mx, lambda m: 1.0); return proj(y, np.ones(me), im)
    if kind == "fixed":
        c = junction_constant(Mx); y, C, sb, im, n = averaged(Mx, lambda m: 1.0)
        s0 = 0.01 * np.median(np.abs(y[im])); return proj(y, 1 / (c["c0"] ** 2 * y ** 2 + s0 ** 2 + sb ** 2), im)
    if kind == "repeat":
        a2, b2 = pooled_repeat(Mx); y, C, sb, im, n = averaged(Mx, lambda m: 1.0); return proj(y, 1 / (a2 * y ** 2 + b2), im)
    if kind in ("constant", "cv", "alignment"):
        mod = junction_constant(Mx) if kind == "constant" else fit_reliability(G, Mx, term="cv" if kind == "cv" else "none")[0]
        return np.asarray(reconcile_dc(G, mod, Mx, dom=dom, n_starts=0, n_shuffle=0)["Qp"])
    if kind == "abs+rel":
        A, B_, S0 = fit_absrel(Mx)
        recs = defaultdict(list)
        for (t, e), m in Mx.items():
            recs[e].append(m)
        y = np.zeros(me); s0v = np.zeros(me); im = np.zeros(me, bool)
        for e, r in recs.items():
            i = ix.get(e)
            if i is None:
                continue
            q = np.array([m["Q"] for m in r]); s = np.array([m["s_abs"] for m in r]); w = 1 / (A ** 2 * q ** 2 + B_ ** 2 * s ** 2 + S0 ** 2); a = w / w.sum()
            y[i] = a @ q; s0v[i] = np.sqrt(S0 ** 2 + B_ ** 2 * (a @ s ** 2) + (a @ (q - y[i]) ** 2 if len(r) > 1 else 0.0)); im[i] = True
        Q0 = proj(y, np.ones(me), im); mm = float(np.median(1 / (A ** 2 * Q0[im] ** 2 + s0v[im] ** 2)))
        return ql_solve(Q0, y, np.full(me, A), im, s0v, mm, dom)[0]
    raise ValueError(kind)


def score(Q, held, acc):
    for e, m in held:
        i = ix.get(tuple(e))
        if i is None or m["Q"] == 0 or not np.isfinite(Q[i]):
            continue
        acc[0].append(abs(np.log(max(abs(Q[i]), 1e-12) / abs(m["Q"])))); acc[1].append(np.sign(Q[i]) == np.sign(m["Q"]))


KINDS = ["average", "uniform", "fixed", "repeat", "constant", "cv", "alignment", "abs+rel"]
folds = list(repeat_folds(M, n_fold=5, seed=0)); usable = sorted({e for (t, e) in M if e in ix})
draws = {p: [{usable[k] for k in np.random.default_rng(10 + r).choice(len(usable), int(p * len(usable)), replace=False)} for r in range(3)]
         for p in (0.1, 0.25)}
out = dict(n_recordings=len(M), n_vessels=len(usable), variants=[])
for kind in KINDS:
    Q = solve(M, kind)
    if kind == "average":
        L = R = np.nan
    else:
        fb = emergent(Q, dom); L, R = fb[inj][sd == "L"].sum(), fb[inj][sd == "R"].sum()
    row = dict(kind=kind, inflow=float(L + R), left=float(L / (L + R)) if kind != "average" else None)
    acc = ([], [])
    for Mf, held in folds:
        score(solve(Mf, kind), held, acc)
    row["loro"] = dict(fold=float(np.exp(np.median(acc[0]))), direction=float(np.mean(acc[1])), n=len(acc[0]))
    for p in (0.1, 0.25):
        acc = ([], [])
        for hide in draws[p]:
            score(solve({k: m for k, m in M.items() if k[1] not in hide}, kind), [(e, m) for (t, e), m in M.items() if e in hide], acc)
        row[f"withheld_{int(p * 100)}"] = dict(fold=float(np.exp(np.median(acc[0]))), direction=float(np.mean(acc[1])), n=len(acc[0]))
    out["variants"].append(row)
    log(f"{kind:10s} inflow {row['inflow']:.2f} left {row['left'] if row['left'] is None else round(100 * row['left'], 1)} | "
        f"LORO ×{row['loro']['fold']:.3f} {row['loro']['direction']:.1%} | 10 % ×{row['withheld_10']['fold']:.3f} "
        f"{row['withheld_10']['direction']:.1%} | 25 % ×{row['withheld_25']['fold']:.3f} {row['withheld_25']['direction']:.1%}")
(P.RESULTS / "dc_uncertainty_models.json").write_text(json.dumps(out, indent=1))
log(f"wrote {P.RESULTS / 'dc_uncertainty_models.json'}")
