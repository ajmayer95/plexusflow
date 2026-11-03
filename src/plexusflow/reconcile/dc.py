"""Full DC reconciliation on the carved network.

1. Per edge the Kloosterman-valid measurement with the smallest reliability c (winner), Q signed u→v.
   Kloosterman-failed / unmeasured edges are unimaged.
2. Q' minimises the quasi-likelihood of the measured flows under exact interior conservation,

       Q' = argmin_q  Σ_imaged ℓ_k(q_k) / m  +  (ε/2) Σ_unimaged q_k²      s.t.  B_I q = 0,
       ℓ_k(q) = ∫_{y_k}^{q} (u − y_k) / (c_k² u² + σ0²) du      (closed form),

   i.e. the measurement variance c²Q² + σ0² is evaluated at the reconciled flow (Wedderburn 1974 quasi-likelihood; data
   reconciliation in the sense of Crowe 1983).  σ0 = 0.01·median|Q_m|; ε = EPS_UNIMAGED is a weak ridge that fixes
   circulations conservation cannot determine (the structural null space ker[B_I; M]: loops of unimaged vessels); m =
   median 1/V at the start makes ε relative to a typical measured weight.  The objective is not convex (ℓ is concave
   beyond q ≈ 2y), so the start is part of the estimator: the uniform-weight conservative projection of the data.
   Solver: projected Newton — each step is one weighted conservative projection with a positive diagonal model of the
   Hessian, followed by an Armijo backtracking line search on the objective — so the objective decreases monotonically
   and the iteration stops at a constrained stationary point (no limit cycle, no averaging).
   Sensitivity: n_starts further solves from perturbed starts (data × (1 + 0.3 N(0, 1)), fixed seeds); Qp_sd = spread;
   `unstable` = a vessel whose direction or magnitude (> stable_rtol) depends on the start.
3. Emergent boundary f = B Q'; passive comparison with K = r⁴/ℓ, consensus radius √(R_graph · median R_kl).

Before 2026-11-03 step 2 was a damped iteratively re-weighted projection whose iterates settle into a small limit cycle
(21S, 27S); Q' was the average of the last 100 iterates over 8 perturbed runs.  Both estimators have the same
stationary equations ((q − y)/V(q) is the IRLS correction term) and validate identically against independent repeat
recordings (scripts/dc_validate_weighting.py); the explicit objective replaced the cycle average.
"""
from __future__ import annotations
import numpy as np
from .reliability import c_of
from .network import carved_domain, project, emergent, passive, metrics, EPS_UNIMAGED

PX_UM = 1.7


def ql_objective(y, C, imaged, s0, m, eps=EPS_UNIMAGED):
    """Objective, gradient and (positive model, exact) diagonal Hessian of the DC quasi-likelihood (see module doc)."""
    c2 = C ** 2
    F = lambda u: np.log(c2 * u ** 2 + s0 ** 2) / (2 * c2) - y * np.arctan(C * u / s0) / (C * s0)
    Fy = F(y)

    def L(Q):
        return float(np.sum((F(Q) - Fy)[imaged]) / m + 0.5 * eps * np.sum(Q[~imaged] ** 2))

    def grad(Q):
        return np.where(imaged, (Q - y) / (c2 * Q ** 2 + s0 ** 2) / m, eps * Q)

    def hess(Q):
        V = c2 * Q ** 2 + s0 ** 2; h = (s0 ** 2 - c2 * Q ** 2 + 2 * c2 * Q * y) / V ** 2
        return np.where(imaged, np.maximum(h, 0.5 / V) / m, eps), np.where(imaged, h / m, eps)
    return L, grad, hess


def ql_solve(Q, y, C, imaged, s0, m, dom, tol=1e-10, max_iter=500):
    """Projected Newton with Armijo backtracking from the conservative start Q → (Q', info)."""
    L, grad, hess = ql_objective(y, C, imaged, s0, m); f = L(Q); converged = False; it = 0
    for it in range(1, max_iter + 1):
        g = grad(Q); G, _ = hess(Q); d = project(Q - g / G, G, dom) - Q; slope = float(g @ d)
        if slope >= 0:                                       # no descent direction left: stationary to machine precision
            converged = True; break
        a = 1.0
        while L(Q + a * d) > f + 1e-4 * a * slope and a > 1e-10:
            a *= 0.5
        Q = Q + a * d; fn = L(Q); rel = np.linalg.norm(a * d) / np.linalg.norm(Q)
        if rel < tol and f - fn <= 1e-14 * abs(f):
            f = fn; converged = True; break
        f = fn
    _, h = hess(Q)
    return Q, dict(objective=f, n_iter=it, converged=converged, n_concave=int(np.sum(imaged & (h < 0))), m=m)


def reconcile_dc(G, model, M, dom=None, tol=1e-10, max_iter=500, n_shuffle=10, seed=0, n_starts=4, start_noise=0.3, stable_rtol=0.10):
    dom = dom or carved_domain(G)
    win = {}
    for (t, e), m in M.items():
        c = c_of(model, m)
        if e not in win or c < win[e][1]:
            win[e] = (m, c)
    sub = dom["edges"]; me = len(sub)
    Qm = np.zeros(me); C = np.full(me, np.nan); imaged = np.zeros(me, bool); tile_w = np.full(me, -1)
    for i, e in enumerate(sub):
        if e in win:
            m, c = win[e]; Qm[i], C[i], imaged[i], tile_w[i] = m["Q"], c, True, m["tile"]
    s0 = 0.01 * np.median(np.abs(Qm[imaged])); Cw = np.where(imaged, C, 1.0)
    start = lambda q: project(q, np.where(imaged, 1.0, EPS_UNIMAGED), dom)          # uniform-weight conservative projection
    Q0 = start(Qm); m = float(np.median(1 / (Cw[imaged] ** 2 * Q0[imaged] ** 2 + s0 ** 2)))
    Qp, info = ql_solve(Q0, Qm, Cw, imaged, s0, m, dom, tol=tol, max_iter=max_iter)
    rng = np.random.default_rng(seed); runs = [Qp]
    for k in range(n_starts):
        runs.append(ql_solve(start(Qm * (1 + start_noise * rng.standard_normal(me))), Qm, Cw, imaged, s0, m, dom, tol=tol, max_iter=max_iter)[0])
    E = np.array(runs); Qp_sd = E.std(axis=0) if n_starts else np.zeros(me)
    unstable = ((np.any(E > 0, axis=0) & np.any(E < 0, axis=0)) | ((E.max(0) - E.min(0)) > stable_rtol * np.maximum(np.abs(Qp), 1e-12))) \
        if n_starts else np.zeros(me, bool)
    n_it, converged = info["n_iter"], info["converged"]
    fb = emergent(Qp, dom)
    rg = np.array([G.edges[e]["radius"] for e in sub], float)
    def _median_rkl(e):
        r = [m["R_kl_um"] / PX_UM for m in G.edges[e].get("measurements_piv") or [] if m.get("R_kl_um") and np.isfinite(m["R_kl_um"])]
        return float(np.median(r)) if r else np.nan
    rt = np.array([_median_rkl(e) for e in sub])
    r = np.sqrt(rg * np.where(np.isfinite(rt), rt, rg)); L = np.array([G.edges[e]["length"] for e in sub], float)
    K = r ** 4 / L; Qs = passive(K, fb, dom)
    met = metrics(Qm, Qp, Qs, fb, dom, imaged)
    rng = np.random.default_rng(seed)
    shuf = np.mean([metrics(Qm, Qp, passive(rng.permutation(K), fb, dom), fb, dom, imaged)["corr_lin"] for _ in range(n_shuffle)])
    return dict(model=model, edges=sub, nodes=dom["nodes"], Qm=Qm, Qp=Qp, Qs=Qs, fb=fb,
                sigma=np.sqrt(Cw ** 2 * Qp ** 2 + s0 ** 2), c=C, imaged=imaged, tile=tile_w, radius=r, length=L,
                inj=dom["inj"], out=dom["out"], side=dom["side"], metrics=met, shuffled_corr=shuf,
                n_iter=n_it, sigma0_floor=s0, dom=dom, estimator="quasi-likelihood, projected Newton (2026-11-03)",
                objective=info["objective"], objective_m=m, n_concave=info["n_concave"],
                converged=converged, Qp_sd=Qp_sd, unstable=unstable, n_starts=n_starts)
