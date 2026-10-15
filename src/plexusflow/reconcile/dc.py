"""Full DC reconciliation on the carved network.

1. Per edge the Kloosterman-valid measurement with the smallest reliability c (winner), Q signed u→v.
   Kloosterman-failed / unmeasured edges are unimaged (weight EPS_UNIMAGED).
2. Q' = argmin Σ w (Q − Q_m)²  s.t. B_I Q = 0,  w = 1 / (c² |Q'|² + σ0²),  σ0 = 0.01·median|Q_m|,
   iterated from uniform weights with damping 0.3 until ‖ΔQ'‖/‖Q'‖ < 1e-6 (σ uses the RECONCILED |Q'|:
   weighting by |Q_m| over-trusts measurements that came out low).
3. Emergent boundary f = B Q'; passive comparison with K = r⁴/ℓ, consensus radius √(R_graph · median R_kl).
"""
from __future__ import annotations
import numpy as np
from .reliability import c_of
from .network import carved_domain, project, emergent, passive, metrics, EPS_UNIMAGED

PX_UM = 1.7


def reconcile_dc(G, model, M, dom=None, damping=0.3, tol=1e-6, max_iter=500, n_shuffle=10, seed=0, n_avg=100,
                 n_check=500, stable_rtol=0.02):
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
    s0 = 0.01 * np.median(np.abs(Qm[imaged])); Cw = np.where(imaged, C, 1.0); ref = None; n_it = 0
    from collections import deque
    tail = deque(maxlen=n_avg); converged = False
    for n_it in range(1, max_iter + 1):
        w = np.ones_like(Qm) if ref is None else 1 / (Cw ** 2 * ref ** 2 + s0 ** 2)
        w = np.where(imaged, w / np.median(w[imaged]), EPS_UNIMAGED)
        new = project(Qm, w, dom)
        if ref is None:
            ref = new; continue
        upd = damping * new + (1 - damping) * ref; ch = np.linalg.norm(upd - ref) / np.linalg.norm(upd); ref = upd
        tail.append(ref)
        if ch < tol:
            converged = True; break
    # The reweighting can settle into a limit cycle instead of converging (weights ∝ 1/Q'² let a vessel whose flow passes
    # near zero pin itself there, which moves its neighbours, and so on); the iterate at the cap then depends on rounding.
    # Report the average over the last n_avg iterates (still exactly conservative: a convex combination of conservative
    # fields) and flag vessels whose direction changes within that window.
    T = np.array(tail)
    Qp = ref if converged else T.mean(axis=0)
    dir_unstable = np.zeros(me, bool) if converged else (np.any(T > 0, axis=0) & np.any(T < 0, axis=0))
    cycle_sd = np.zeros(me) if converged else T.std(axis=0)
    # Reproducibility check: continue n_check more iterations and average again; vessels whose averaged flow moves by
    # > stable_rtol (relative) or changes direction are weakly determined (mostly unmeasured loop flows and near-zero
    # vessels) — flagged `unstable`; their values can differ across platforms / iteration counts.
    unstable = dir_unstable.copy()
    if not converged and n_check > 0:
        tail2 = deque(maxlen=n_avg)
        for _ in range(n_check):
            w = 1 / (Cw ** 2 * ref ** 2 + s0 ** 2); w = np.where(imaged, w / np.median(w[imaged]), EPS_UNIMAGED)
            ref = damping * project(Qm, w, dom) + (1 - damping) * ref; tail2.append(ref)
        Q2 = np.array(tail2).mean(axis=0); scale = np.maximum(np.abs(Qp), 1e-6)
        unstable |= (np.abs(Q2 - Qp) / scale > stable_rtol) | (np.sign(Q2) != np.sign(Qp))
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
                n_iter=n_it, sigma0_floor=s0, dom=dom,
                converged=converged, dir_unstable=dir_unstable, Qp_cycle_sd=cycle_sd, unstable=unstable)
