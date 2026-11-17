"""Full DC reconciliation on the carved network (final method, frozen 2026-11-12).

0. Network: network.carved_domain — open-boundary regions removed with free inflow boundaries, topology-uncertain regions
   removed as mass-conserving black boxes (one conservation row per region; reconcile.regions).  Both are graph tags set
   in the workbench viewer; a stage without tags reconstructs on the whole network minus the DA anchors.

1. Per edge the 1/c²-weighted average of its Kloosterman-valid recordings with weighted-RMS reliability (vessel_input;
   before 2026-11-10 the single recording with the smallest c), Q signed u→v.  Kloosterman-failed / unmeasured edges are unimaged.
2. Q' minimises the quasi-likelihood of the measured flows under exact interior conservation,

       Q' = argmin_q  Σ_imaged ℓ_k(q_k) / m  +  (ε/2) Σ_unimaged q_k²      s.t.  B_I q = 0,
       ℓ_k(q) = ∫_{y_k}^{q} (u − y_k) / (c_k² u² + σ0²) du      (closed form),

   i.e. the measurement variance c²Q′² + σ0,k² (σ0,k² = σ0² + s_between,k², the disagreement of vessel k's recordings —
   vessel_input) is evaluated at the reconciled flow (Wedderburn 1974 quasi-likelihood; data
   reconciliation in the sense of Crowe 1983).  σ0 = 0.01·median|Q_m|; ε = EPS_UNIMAGED is a weak ridge that fixes
   circulations conservation cannot determine (the structural null space ker[B_I; M]: loops of unimaged vessels); m =
   median 1/V at the start makes ε relative to a typical measured weight.  The objective is not convex (ℓ is concave
   beyond q ≈ 2y), so the start is part of the estimator: the uniform-weight conservative projection of the data.
   Solver: projected Newton — each step is one weighted conservative projection with a positive diagonal model of the
   Hessian, followed by an Armijo backtracking line search on the objective — so the objective decreases monotonically
   and the iteration stops at a constrained stationary point (no limit cycle, no averaging).
   Sensitivity: n_starts further solves from perturbed starts (data × (1 + 0.3 N(0, 1)), fixed seeds); Qp_sd = spread;
   `unstable` = a vessel whose direction or magnitude (> stable_rtol) depends on the start.
3. Emergent boundary f = B Q'; passive comparison with K = r⁴/ℓ.  ``radius`` picks r (also handed to the AC model):
   "tube" (default since 2026-11-17) = the uncalibrated tube-interior segmentation radius (reconcile/radius.tube_radii on
   <stage>/mask/mask.tif): median walled half-width for resistance, volume-equivalent √⟨a²⟩ for compliance (returned as
   ``radius_compliance`` for the AC model); "consensus" = √(R_graph · median R_kl) (sensitivity analysis; default before
   2026-11-17); "anatomical" = the calibrated segmentation radius (results/anatomical_radius.pkl); or an array (used for
   both).  The flux estimator is not affected: it keeps the free-radius Kloosterman fit.

Before 2026-11-03 step 2 was a damped iteratively re-weighted projection whose iterates settle into a small limit cycle
(21S, 27S); Q' was the average of the last 100 iterates over 8 perturbed runs.  Both estimators have the same
stationary equations ((q − y)/V(q) is the IRLS correction term) and validate identically against independent repeat
recordings (scripts/dc_validate_weighting.py); the explicit objective replaced the cycle average.
"""
from __future__ import annotations
import numpy as np
from .reliability import c_of
from .network import carved_domain, project, emergent, passive, metrics, contract_short_unimaged, EPS_UNIMAGED

PX_UM = 1.7


PER_VESSEL = "average"      # since 2026-11-10 (was "winner"): see vessel_input


def vessel_input(recs, per_vessel=PER_VESSEL):
    """One vessel's (y, c, s_between) from its recordings [(measurement, c), …].

    "average" (production since 2026-11-10): y = Σ w y_t / Σ w with w = 1/c², and the vessel's reliability is the
    weighted-RMS quality of its recordings, c = √(Σ w c² / Σ w) — the typical quality of what was averaged, with no extra
    credit for the number of recordings (the recordings are different moments, not independent samples of one flow).
    Recordings of one vessel come from tiles acquired at different times; the reconstruction targets a conservation-
    consistent reference field summarising the mosaic, so no recording time is privileged.  "winner": the smallest-c
    recording (production before 2026-11-10).
    Evidence (sandbox/agg_compare.py, 21S / 27S): held-out error of the reconciled field vs winner −0.011 [−0.022, +0.003] /
    +0.016 [−0.019, +0.052] (trusted, tile bootstrap); junction calibration RMS z 1.07 / 1.02 (ideal 1) vs 1.43 / 1.32 for
    the independent-repeats combination (Σ 1/c²)^(−1/2) and 1.30 / 1.14 for winner.  Equal weights were worse (27S +0.07);
    1/(c² + τ²) weights and adding the RELATIVE spread to c gave no gain.
    s_between = √(Σ α (y_t − y)²), α = w/Σw: the disagreement between the recordings, an absolute flux scale.  It enters the
    reconciliation variance as a per-vessel floor, σ² = c²Q′² + σ0² + s_between² (every multiply recorded vessel; 0 for one
    recording).  Not calibrated on junctions — a regularisation scale for incompatible observations: without it a vessel
    whose recordings disagree in sign averages to ≈ 0 and gets 4–5× the median weight (21S / 27S: 4.9× / 3.8× → 0.9× / 0.5×);
    held-out error unchanged (21S all −0.021 [−0.045, 0.000], 27S +0.010 [−0.026, +0.049] trusted vs winner)."""
    if per_vessel == "winner" or len(recs) == 1:
        m, c = min(recs, key=lambda x: x[1]); return float(m["Q"]), float(c), 0.0
    assert per_vessel == "average", per_vessel
    cs = np.array([c for _, c in recs]); a = (1 / cs ** 2) / np.sum(1 / cs ** 2); q = np.array([m["Q"] for m, _ in recs]); y = float(a @ q)
    return y, float(np.sqrt(a @ cs ** 2)), float(np.sqrt(a @ (q - y) ** 2))


def repeat_folds(M, n_fold=5, seed=0):
    """Independent references for averaged inputs: per fold, one random recording of every vessel recorded in ≥ 2 tiles is
    held out.  Yields (M without the held recordings, [(edge, held measurement)]).  With per_vessel="average" every recording
    enters the input, so a vessel's other recordings are no longer independent of its reconstructed flow."""
    by = {}
    for k in M:
        by.setdefault(k[1], []).append(k)
    rng = np.random.default_rng(seed)
    for _ in range(n_fold):
        held = {e: ks[int(rng.integers(len(ks)))] for e, ks in by.items() if len(ks) >= 2}
        hs = set(held.values())
        yield {k: m for k, m in M.items() if k not in hs}, [(e, M[k]) for e, k in held.items()]


def anatomical_radius_array(edges, path=None):
    """Calibrated anatomical radius (per-vessel median of R(s)) in mosaic px for ``edges``, from scripts/anatomical_radius.py."""
    import pickle
    from pathlib import Path
    from .. import paths
    A = pickle.loads(Path(path or paths.RESULTS / "anatomical_radius.pkl").read_bytes()); idx = {e: i for i, e in enumerate(A["edges"])}
    return np.array([A["median"][idx[e] if e in idx else idx[e[::-1]]] for e in edges], float)


_TUBE_CACHE = {}


def tube_radius_arrays(G, edges, mask=None):
    """(resistance, compliance, profiled) tube-interior radii for ``edges``; mask defaults to <stage>/mask/mask.tif.
    Cached per (graph object, edge list, mask) so repeated reconciliations (withholding folds) profile once."""
    from .radius import tube_radii
    key = (id(G), G.number_of_edges(), tuple(edges), None if mask is None else id(mask))
    if key not in _TUBE_CACHE:
        if mask is None:
            import tifffile
            from .. import paths
            mask = tifffile.imread(str(paths.STAGE / "mask" / "mask.tif")) > 127
        T = tube_radii(G, edges, mask); _TUBE_CACHE.clear(); _TUBE_CACHE[key] = T
    T = _TUBE_CACHE[key]
    return T["resistance"], T["compliance"], T["profiled"]


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


def reconcile_dc(G, model, M, dom=None, topology=True, tol=1e-10, max_iter=500, n_shuffle=10, seed=0, n_starts=4, start_noise=0.3, stable_rtol=0.10,
                 radius="tube", per_vessel=PER_VESSEL, mask=None):
    dom = dom or carved_domain(G, topology=topology)
    recs = {}
    for (t, e), m in M.items():
        recs.setdefault(e, []).append((m, c_of(model, m)))
    sub = dom["edges"]; me = len(sub)
    Qm = np.zeros(me); C = np.full(me, np.nan); imaged = np.zeros(me, bool); tile_w = np.full(me, -1); n_rec = np.zeros(me, int); sb = np.zeros(me)
    for i, e in enumerate(sub):
        if e in recs:
            Qm[i], C[i], sb[i] = vessel_input(recs[e], per_vessel); imaged[i] = True; n_rec[i] = len(recs[e])
            tile_w[i] = min(recs[e], key=lambda x: x[1])[0]["tile"]                  # best recording's tile (labels / figures)
    s0 = 0.01 * np.median(np.abs(Qm[imaged])); Cw = np.where(imaged, C, 1.0)
    s0v = np.sqrt(s0 ** 2 + sb ** 2)                                                # per-vessel floor: σ0 + between-recording spread
    start = lambda q: project(q, np.where(imaged, 1.0, EPS_UNIMAGED), dom)          # uniform-weight conservative projection
    Q0 = start(Qm); m = float(np.median(1 / (Cw[imaged] ** 2 * Q0[imaged] ** 2 + s0 ** 2)))
    Qp, info = ql_solve(Q0, Qm, Cw, imaged, s0v, m, dom, tol=tol, max_iter=max_iter)
    rng = np.random.default_rng(seed); runs = [Qp]
    for k in range(n_starts):
        runs.append(ql_solve(start(Qm * (1 + start_noise * rng.standard_normal(me))), Qm, Cw, imaged, s0v, m, dom, tol=tol, max_iter=max_iter)[0])
    E = np.array(runs); Qp_sd = E.std(axis=0) if n_starts else np.zeros(me)
    unstable = ((np.any(E > 0, axis=0) & np.any(E < 0, axis=0)) | ((E.max(0) - E.min(0)) > stable_rtol * np.maximum(np.abs(Qp), 1e-12))) \
        if n_starts else np.zeros(me, bool)
    n_it, converged = info["n_iter"], info["converged"]
    fb = emergent(Qp, dom)
    f_all = dom["B"] @ Qp; nix = {n: k for k, n in enumerate(dom["nodes"])}
    topo_regions = [dict(n_edges=r["n_edges"], cut=r["cut"], leaky=r["leaky"],
                         net=float(sum(f_all[nix[n]] for n in r["cut"])), throughput=float(0.5 * sum(abs(f_all[nix[n]]) for n in r["cut"])))
                    for r in dom.get("regions", [])]
    rg = np.array([G.edges[e]["radius"] for e in sub], float)
    def _median_rkl(e):
        r = [m["R_kl_um"] / PX_UM for m in G.edges[e].get("measurements_piv") or [] if m.get("R_kl_um") and np.isfinite(m["R_kl_um"])]
        return float(np.median(r)) if r else np.nan
    rt = np.array([_median_rkl(e) for e in sub])
    r_cons = np.sqrt(rg * np.where(np.isfinite(rt), rt, rg)); L = np.array([G.edges[e]["length"] for e in sub], float)
    profiled = np.zeros(me, bool)
    if isinstance(radius, str) and radius == "tube":
        r, rc, profiled = tube_radius_arrays(G, sub, mask)
    elif isinstance(radius, str) and radius == "anatomical":
        r = anatomical_radius_array(sub); rc = r
    elif isinstance(radius, str):
        assert radius == "consensus", radius; r = rc = r_cons
    else:
        r = rc = np.asarray(radius, float)
    K = r ** 4 / L; Qs = passive(K, fb, dom)
    met = metrics(Qm, Qp, Qs, fb, dom, imaged)
    con = contract_short_unimaged(dom, imaged, L, r); short_unimaged = con.pop("short_unimaged"); met["contracted"] = con
    rng = np.random.default_rng(seed)
    shuf = np.mean([metrics(Qm, Qp, passive(rng.permutation(K), fb, dom), fb, dom, imaged)["corr_lin"] for _ in range(n_shuffle)])
    return dict(model=model, edges=sub, nodes=dom["nodes"], Qm=Qm, Qp=Qp, Qs=Qs, fb=fb,
                sigma=np.sqrt(Cw ** 2 * Qp ** 2 + s0v ** 2), c=C, s_between=sb, s0_vessel=s0v, imaged=imaged, tile=tile_w, radius=r, radius_compliance=rc,
                radius_profiled=profiled, length=L,
                inj=dom["inj"], out=dom["out"], side=dom["side"], metrics=met, shuffled_corr=shuf,
                n_iter=n_it, sigma0_floor=s0, dom=dom, estimator="quasi-likelihood, projected Newton (2026-11-03)",
                objective=info["objective"], objective_m=m, n_concave=info["n_concave"],
                converged=converged, Qp_sd=Qp_sd, unstable=unstable, n_starts=n_starts, short_unimaged=short_unimaged,
                radius_model=radius if isinstance(radius, str) else "given", radius_consensus=r_cons, per_vessel=per_vessel, n_rec=n_rec,
                topology_regions=topo_regions, n_topology_edges=int(sum(r["n_edges"] for r in dom.get("regions", []))))
