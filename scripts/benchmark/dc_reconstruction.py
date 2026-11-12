"""Known-truth benchmark of the DC reconstruction on the real 21S network: conservation-only vs passive-resistance-constrained
under measurement noise × radius uncertainty (SI; reviewer request 2026-11-04).

Geometry: the carved network of the stage, consensus radii a* and lengths (results/dc_reconciled_v2.pkl), the real
measurement mask (vessels unmeasured in the experiment stay hidden), the real per-vessel reliability c_k and floor s0.
Truth per realisation: Q* = passive Poiseuille flow (conductance a*⁴/ℓ) driven at the boundary by the reconstructed
boundary fluxes × exp(0.3 N(0,1)) (sign kept), veins rescaled to balance — a different, purely passive truth each time
(the reconciled field itself is never the truth).
Measurements: y_k = Q*_k + κ √(c_k² Q*_k² + s0²) N(0,1) on the experimentally measured vessels.
Reconstructions (production code, plexusflow.reconcile):
  kcl_ql       conservation only, quasi-likelihood (production: projected Newton from the uniform projection)
  kcl_relative conservation only, fixed weights 1/(c² y² + s0²)
  kcl_uniform  conservation only, equal weights
  pas_ql       passive-resistance-constrained, same quasi-likelihood, radii â = a*(1 + δ)
  pas_relative passive-resistance-constrained, fixed weights 1/(c² y² + s0²)
Radius error δ: independent N(0, σ_a²) per vessel ('vessel'), or tile-correlated δ = δ_tile + δ_vessel with σ_tile = 0.1
('tile+vessel'; tile = the vessel's measurement tile, or a nearby one).
Scores vs Q*, separately for measured vessels and hidden vessels — hidden split into those determined by conservation given
the measured ones ('hidden_det') and those on loops of unmeasured vessels (structural null space ker[B_int; M], 'hidden_loop')
— over vessels with |Q*| ≥ its 10th percentile: median |log(|Q|/|Q*|)| and direction agreement.
Crossover: radius error at which E(pas_ql) − E(kcl_ql), averaged over realisations, turns positive (linear interpolation);
95 % interval from a bootstrap over realisations (paired across radius errors).
Radius disagreement (radius_error_estimate): spread of ln(r_graph / r_KL) — inter-method disagreement, not an absolute error;
it equals the consensus-radius error ×2 only if the two errors are independent, and is blind to errors both methods share.
Writes $PLEXUSFLOW_RESULTS/dc_synthetic_benchmark.json.
Usage: python scripts/benchmark/dc_reconstruction.py [--reps 20] [--jobs 8]
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.optimize import minimize
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.reconcile.network import carved_domain, project, passive, EPS_UNIMAGED
from plexusflow.reconcile.dc import ql_objective, ql_solve

KAPPAS = (0.25, 0.5, 1.0, 2.0)
SIGMA_A = (0.0, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.3)
ERR_MODES = ("vessel", "tile+vessel")
METHODS = ("kcl_ql", "kcl_relative", "kcl_uniform", "pas_ql", "pas_relative")


@lru_cache(maxsize=1)
def ctx():
    G = paths.load_graph(); dom = carved_domain(G); D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes())
    assert list(dom["edges"]) == [tuple(e) for e in D["edges"]]
    im = D["imaged"]; C = np.where(im, D["c"], 1.0)
    tile = np.asarray(D["tile"]).copy()                                      # unimaged vessels: tile of a measured neighbour
    B = dom["B"].tocsr(); Bt = dom["B"].T.tocsr()
    for _ in range(6):
        miss = np.flatnonzero(tile < 0)
        if not len(miss):
            break
        for e in miss:
            nb = {j for n in Bt[e].indices for j in B[n].indices} - {e}; t = [tile[j] for j in nb if tile[j] >= 0]
            if t:
                tile[e] = t[0]
    nodes = dom["nodes"]; bnd = set(np.asarray(dom["bnd"]).tolist()); ix = {n: i for i, n in enumerate(nodes)}
    import networkx as nx
    H = nx.MultiGraph(); loop = np.zeros(len(im), bool)
    for k, (u, v) in enumerate(dom["edges"]):
        if not im[k]:
            H.add_edge(-1 if ix[u] in bnd else ix[u], -1 if ix[v] in bnd else ix[v], key=k)
    simple = nx.Graph(); mult = {}
    for x, y_, k in H.edges(keys=True):
        if x == y_:
            loop[k] = True; continue
        mult.setdefault(tuple(sorted((x, y_))), []).append(k); simple.add_edge(x, y_)
    br = {tuple(sorted(e)) for e in nx.bridges(simple)}
    for key, ks in mult.items():
        if key not in br or len(ks) > 1:
            loop[ks] = True
    return dict(loop=loop, dom=dom, a=np.asarray(D["radius"], float), l=np.asarray(D["length"], float), im=im, C=C, s0=float(D["sigma0_floor"]),
                fb=np.asarray(D["fb"], float), tile=np.where(tile < 0, 0, tile))


def radius_error_estimate(G, dom, px_um=1.7):
    """Per-vessel radius uncertainty from the two independent estimates (graph radius vs median Kloosterman R_kl): robust SD of
    ln(r_graph / r_KL); with independent equal errors the consensus radius √(r_graph r_KL) has σ ≈ SD/2.  Errors shared by
    both methods cancel in the ratio, so this is a lower bound."""
    lr = []
    for e in dom["edges"]:
        rg = G.edges[e]["radius"]; rk = [m["R_kl_um"] / px_um for m in G.edges[e].get("measurements_piv") or [] if m.get("R_kl_um") and np.isfinite(m["R_kl_um"])]
        if rk and rg > 0:
            lr.append(np.log(rg / np.median(rk)))
    lr = np.array(lr); sd = 1.4826 * float(np.median(np.abs(lr - np.median(lr))))
    return dict(n=int(len(lr)), median_log_ratio=float(np.median(lr)), robust_sd_log_ratio=sd, sigma_consensus=sd / 2, sigma_single=sd / np.sqrt(2))


def passive_basis(dom, K):
    B = dom["B"].tocsc(); bnd = np.asarray(dom["bnd"]); I = np.setdiff1d(np.arange(B.shape[0]), bnd)
    Lap = (B @ sp.diags(K) @ B.T).tocsc(); pB = np.eye(len(bnd))[:, 1:]
    pI = -spla.splu(Lap[I][:, I].tocsc()).solve(Lap[I][:, bnd].toarray() @ pB)
    Pn = np.zeros((B.shape[0], pB.shape[1])); Pn[I] = pI; Pn[bnd] = pB
    return np.asarray(sp.diags(K) @ (B.T @ Pn))


def passive_fits(A, y, C, im, s0, m):
    w = np.sqrt(1 / (C[im] ** 2 * y[im] ** 2 + s0 ** 2)); p0, *_ = np.linalg.lstsq(A[im] * w[:, None], y[im] * w, rcond=None)
    L, grad, _ = ql_objective(y, C, im, s0, m)
    r = minimize(lambda p: (L(A @ p), A.T @ grad(A @ p)), p0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=50000, maxfun=200000, ftol=1e-16, gtol=1e-14, maxcor=50))
    return A @ r.x, A @ p0


def score(Q, Qt, sel):
    a = np.abs(Qt); v = sel & (a >= np.percentile(a[sel], 10))
    return (float(np.median(np.abs(np.log(np.maximum(np.abs(Q[v]), 1e-12) / a[v])))), float(np.mean(np.sign(Q[v]) == np.sign(Qt[v]))))


def realisation(kappa, rep):
    X = ctx(); dom, a, l, im, C, s0 = X["dom"], X["a"], X["l"], X["im"], X["C"], X["s0"]
    rng = np.random.default_rng(1000 * rep + int(100 * kappa))
    fb = X["fb"].copy(); inj, out = np.asarray(dom["inj"]), np.asarray(dom["out"])
    fb[inj] *= np.exp(0.3 * rng.standard_normal(len(inj))); fb[out] *= -fb[inj].sum() / fb[out].sum()
    Qt = passive(a ** 4 / l, fb, dom)
    y = np.where(im, Qt + kappa * np.sqrt(C ** 2 * Qt ** 2 + s0 ** 2) * rng.standard_normal(len(Qt)), 0.0)
    unif = np.where(im, 1.0, EPS_UNIMAGED); Q0 = project(y, unif, dom); m = float(np.median(1 / (C[im] ** 2 * Q0[im] ** 2 + s0 ** 2)))
    wr = 1 / (C ** 2 * y ** 2 + s0 ** 2); wr = np.where(im, wr / np.median(wr[im]), EPS_UNIMAGED)
    kcl = {"kcl_ql": ql_solve(Q0, y, C, im, s0, m, dom)[0], "kcl_relative": project(y, wr, dom), "kcl_uniform": Q0}
    out_rows = []
    for mode in ERR_MODES:
        for sa in SIGMA_A:
            if mode == "tile+vessel" and sa == 0.0:
                continue
            d = sa * rng.standard_normal(len(a))
            if mode == "tile+vessel":
                tt = np.unique(X["tile"]); dt = dict(zip(tt, 0.1 * rng.standard_normal(len(tt)))); d = d + np.array([dt[t] for t in X["tile"]])
            A = passive_basis(dom, (a * np.maximum(1 + d, 0.2)) ** 4 / l); pq, pr = passive_fits(A, y, C, im, s0, m)
            F = dict(kcl, pas_ql=pq, pas_relative=pr)
            for meth, Q in F.items():
                for lab, sel in (("measured", im), ("hidden", ~im), ("hidden_det", ~im & ~X["loop"]), ("hidden_loop", X["loop"])):
                    e, dr = score(Q, Qt, sel)
                    out_rows.append(dict(kappa=kappa, rep=rep, mode=mode, sigma_a=sa, method=meth, set=lab, err=e, dir=dr))
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    X = ctx(); print(f"{paths.STAGE.name}: {len(X['im'])} vessels, {X['im'].sum()} measured; {len(KAPPAS)} κ × {a.reps} realisations", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import dc_reconstruction as _mod   # importable for the worker processes
    rows = [r for rs in Parallel(n_jobs=a.jobs, verbose=5)(delayed(_mod.realisation)(k, r) for k in KAPPAS for r in range(a.reps)) for r in rs]
    rng = np.random.default_rng(0); crossover = {}
    idx = {(r["mode"], r["kappa"], r["sigma_a"], r["rep"], r["set"], r["method"]): r["err"] for r in rows}
    for mode in ERR_MODES:
        S_ = [s for s in SIGMA_A if not (mode == "tile+vessel" and s == 0.0)]
        for k in KAPPAS:
            for st in ("measured", "hidden", "hidden_det", "hidden_loop"):
                d = np.array([[idx[(mode, k, s, rp, st, "pas_ql")] - idx[(mode, k, s, rp, st, "kcl_ql")] for s in S_]
                              for rp in range(a.reps)])                              # reps × σ_a, paired ΔE
                def cross(m_):
                    if m_[0] > 0:
                        return S_[0]
                    j = next((i for i in range(1, len(m_)) if m_[i] > 0), None)
                    return float(np.nan) if j is None else float(S_[j - 1] + (S_[j] - S_[j - 1]) * (-m_[j - 1]) / (m_[j] - m_[j - 1]))
                bs = [cross(d[rng.integers(0, a.reps, a.reps)].mean(0)) for _ in range(1000)]
                crossover[f"{mode}|{k}|{st}"] = dict(estimate=cross(d.mean(0)), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                                     frac_none=float(np.mean(np.isnan(bs))), mean_dE=d.mean(0).tolist(), sigma_a=S_)
    summ = {}
    for r in rows:
        summ.setdefault((r["mode"], r["kappa"], r["sigma_a"], r["method"], r["set"]), []).append((r["err"], r["dir"]))
    table = [dict(mode=k[0], kappa=k[1], sigma_a=k[2], method=k[3], set=k[4], err_median=float(np.median([v[0] for v in vs])),
                  err_iqr=[float(np.percentile([v[0] for v in vs], q)) for q in (25, 75)], dir_mean=float(np.mean([v[1] for v in vs])), n=len(vs))
             for k, vs in summ.items()]
    G = paths.load_graph(); rerr = radius_error_estimate(G, carved_domain(G))
    print(f"inter-method radius disagreement, graph vs Kloosterman ({rerr['n']} vessels): robust SD of ln ratio {rerr['robust_sd_log_ratio']:.3f} "
          f"(= consensus error {rerr['sigma_consensus']:.2f} only if the two errors are independent; median offset {rerr['median_log_ratio']:+.2f}, "
          f"a uniform scale that cancels in passive flow patterns)", flush=True)
    for st in ("measured", "hidden", "hidden_det", "hidden_loop"):
        c_ = crossover[f"vessel|1.0|{st}"]
        print(f"crossover at κ = 1, {st:11s}: σ_a* = {c_['estimate']:.3f} [95 % {c_['ci'][0]:.3f}, {c_['ci'][1]:.3f}]"
              + (f" (no crossover in {100 * c_['frac_none']:.0f} % of bootstraps)" if c_['frac_none'] else ""), flush=True)
    out = dict(stage=paths.STAGE.name, radius_error_estimate=rerr, crossover=crossover, n_loop=int(X["loop"].sum()), kappas=KAPPAS, sigma_a=SIGMA_A, modes=ERR_MODES, methods=METHODS, reps=a.reps, rows=rows, table=table)
    paths.RESULTS.mkdir(parents=True, exist_ok=True); (paths.RESULTS / "dc_synthetic_benchmark.json").write_text(json.dumps(out))
    get = lambda mode, k, s, meth, st: next(t for t in table if (t["mode"], t["kappa"], t["sigma_a"], t["method"], t["set"]) == (mode, k, s, meth, st))
    for st in ("measured", "hidden_det", "hidden_loop"):
        print(f"\n{st} vessels — median |log(Q/Q*)| (direction %), radius error per vessel:")
        print("  κ     " + "".join(f"{m:>16s}" for m in ("kcl_ql", "kcl_relative", "kcl_uniform")) + "".join(f"   pas_ql σa={s:<4}" for s in SIGMA_A))
        for k in KAPPAS:
            s = "".join(f"{get('vessel', k, 0.0, m, st)['err_median']:8.3f} ({100 * get('vessel', k, 0.0, m, st)['dir_mean']:4.1f})" for m in ("kcl_ql", "kcl_relative", "kcl_uniform"))
            s += "".join(f"{get('vessel', k, sa, 'pas_ql', st)['err_median']:9.3f} ({100 * get('vessel', k, sa, 'pas_ql', st)['dir_mean']:4.1f})" for sa in SIGMA_A)
            print(f"  {k:<5}" + s)
    print(f"\nwrote {paths.RESULTS / 'dc_synthetic_benchmark.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
