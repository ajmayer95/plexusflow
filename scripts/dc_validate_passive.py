"""Conservation-only vs passive-resistance-constrained DC reconstruction, scored against independent repeat recordings
(input to SI figure fig6_S_passive).

Passive-resistance-constrained field: every vessel obeys Δp = R q with R = ℓ/r⁴ (consensus radii, as the Fig 7 passive
model), i.e. q = K ⊙ Bᵀp with interior conservation → q = A p_B (boundary pressures, one gauge).  It is fitted with the same
quasi-likelihood objective as the production reconstruction (plexusflow.reconcile.dc), by L-BFGS over p_B started from the
fixed-weight (1/σ² at Q′) least-squares fit; convergence is reported as the projected gradient ‖Aᵀ∇L‖ relative to its start.
This tests the passive resistance model with segmented geometry, not loop pressure balance as such (any scalar pressure
field satisfies that).
Tests (same vessels and folds as scripts/dc_validate_weighting.py): measured vessels vs held-out repeats; hidden vessels
(all measurements of 10 % of the repeat-measured vessels withheld, 5 folds) vs their withheld measurements.  Scores per
reference: |log(|Q|/|h|)| and direction agreement; paired differences (passive − conservation-only) with 95 % intervals
from a vessel-clustered bootstrap; trusted (c < 0.5) and all references.
Writes $PLEXUSFLOW_RESULTS/dc_passive_comparison.json.  ~2 min (21S).
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.optimize import minimize
import plexusflow.paths as P
from plexusflow.reconcile.network import carved_domain, emergent, metrics, passive
from plexusflow.reconcile.reliability import measurement_table, c_of
from plexusflow.reconcile.dc import reconcile_dc, ql_objective, repeat_folds

TRUST = 0.5
T0 = time.time()
log = lambda s: print(f"[+{time.time() - T0:4.0f} s] {s}", flush=True)


def passive_basis(dom, K):
    """A (nE × nB−1): vessel flows of the passive network per unit boundary pressure (first boundary node = gauge)."""
    B = dom["B"].tocsc(); bnd = np.asarray(dom["bnd"]); I = np.setdiff1d(np.arange(B.shape[0]), bnd)
    Lap = (B @ sp.diags(K) @ B.T).tocsc(); pB = np.eye(len(bnd))[:, 1:]
    pI = -spla.splu(Lap[I][:, I].tocsc()).solve(Lap[I][:, bnd].toarray() @ pB)
    Pn = np.zeros((B.shape[0], pB.shape[1])); Pn[I] = pI; Pn[bnd] = pB
    return np.asarray(sp.diags(K) @ (B.T @ Pn))


def passive_fit(D, A, max_iter=200000):
    y, im = D["Qm"], D["imaged"]; C = np.where(im, D["c"], 1.0); s0 = D.get("s0_vessel", D["sigma0_floor"]); Q = D["Qp"]   # per-vessel floor
    s0i = np.broadcast_to(s0, y.shape)[im]                                     # per-vessel floor on the measured vessels
    w = np.sqrt(1 / (C[im] ** 2 * Q[im] ** 2 + s0i ** 2)); p0, *_ = np.linalg.lstsq(A[im] * w[:, None], y[im] * w, rcond=None)
    m = float(np.median(1 / (C[im] ** 2 * Q[im] ** 2 + s0i ** 2))); L, grad, _ = ql_objective(y, C, im, s0, m)
    f = lambda p: (L(A @ p), A.T @ grad(A @ p)); g0 = np.linalg.norm(f(p0)[1])
    r = minimize(f, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=max_iter, maxfun=4 * max_iter, ftol=1e-16, gtol=1e-14, maxcor=50))
    return A @ r.x, dict(converged=bool(r.success), message=str(r.message), n_iter=int(r.nit), rel_proj_grad=float(np.linalg.norm(r.jac) / g0))


def paired(rows, rng, n_boot=1000):
    """rows: (vessel, err_cons, err_pass, dir_cons, dir_pass, c_ref) → per reference set: medians / rates and paired Δ with CI."""
    a = np.array(rows, float); out = {}
    for lab, sel in (("trusted", a[:, 5] < TRUST), ("all", np.ones(len(a), bool))):
        b = a[sel]; ves = np.unique(b[:, 0]); grp = [np.nonzero(b[:, 0] == v)[0] for v in ves]; de, dd = [], []
        for _ in range(n_boot):
            ix = np.concatenate([grp[k] for k in rng.integers(0, len(grp), len(grp))])
            de.append(np.median(b[ix, 2]) - np.median(b[ix, 1])); dd.append(b[ix, 4].mean() - b[ix, 3].mean())
        out[lab] = dict(err_cons=float(np.median(b[:, 1])), err_pass=float(np.median(b[:, 2])), dir_cons=float(b[:, 3].mean()),
                        dir_pass=float(b[:, 4].mean()), d_err=float(np.median(b[:, 2]) - np.median(b[:, 1])), d_err_ci=np.percentile(de, [2.5, 97.5]).tolist(),
                        d_dir=float(b[:, 4].mean() - b[:, 3].mean()), d_dir_ci=np.percentile(dd, [2.5, 97.5]).tolist(), n=int(len(b)), n_vessels=int(len(ves)))
    return out


def rows_for(Qc, Qp_, refs, model):
    out = []
    for i, m in refs:
        h = m["Q"]
        out.append((i, abs(np.log(max(abs(Qc[i]), 1e-12) / abs(h))), abs(np.log(max(abs(Qp_[i]), 1e-12) / abs(h))),
                    float(np.sign(Qc[i]) == np.sign(h)), float(np.sign(Qp_[i]) == np.sign(h)), c_of(model, m)))
    return out


def main(n_fold=5, frac=0.10, seed=0):
    G = P.load_graph(); M = measurement_table(G); model = json.loads((P.RESULTS / "reliability_model_v1.json").read_text())
    D = pickle.loads((P.RESULTS / "dc_reconciled_v2.pkl").read_bytes()); dom = carved_domain(G)
    E = list(dom["edges"]); assert E == [tuple(e) for e in D["edges"]]; eix = {e: i for i, e in enumerate(E)}
    K = D["radius"] ** 4 / D["length"]; A = passive_basis(dom, K); rng = np.random.default_rng(seed)
    Qpas, conv = passive_fit(D, A)
    log(f"{P.STAGE.name}: {A.shape[1]} boundary pressures; passive fit {conv['message']} after {conv['n_iter']} iterations, "
        f"projected gradient {conv['rel_proj_grad']:.1e} of its start")
    by_e = {}
    for (t, e), m in M.items():
        if e in eix:
            by_e.setdefault(e, []).append(m)
    if D.get("per_vessel", "winner") == "average":           # every recording is input: hold one out per repeat-measured vessel
        mrows = []
        for Mf, held in repeat_folds(M, n_fold, seed):
            Df = reconcile_dc(G, model, Mf, dom=dom, n_starts=0, n_shuffle=1); Qpf, _ = passive_fit(Df, A)
            mrows += rows_for(Df["Qp"], Qpf, [(eix[e], m) for e, m in held if e in eix and abs(m["Q"]) > 0], model)
        measured = paired(mrows, rng)
    else:
        refs = [(eix[e], m) for e, ms in by_e.items() for m in ms if m["tile"] != D["tile"][eix[e]] and abs(m["Q"]) > 0]
        measured = paired(rows_for(D["Qp"], Qpas, refs, model), rng)
    totals = {}
    for lab, Q in (("conservation_only", D["Qp"]), ("passive", Qpas)):
        fb = emergent(Q, dom); mt = metrics(D["Qm"], Q, passive(K, fb, dom), fb, dom, D["imaged"]); big = D["imaged"] & (np.abs(D["Qm"]) > 1e-3)
        totals[lab] = dict(inflow=mt["total_in"], inj_L=mt["inj_L"], inj_R=mt["inj_R"], flips=float(np.mean(np.sign(Q[big]) != np.sign(D["Qm"][big]))),
                           move=float(np.median(np.abs(Q[big] - D["Qm"][big]) / np.abs(D["Qm"][big]))))
    multi = sorted(e for e, ms in by_e.items() if len(ms) >= 2); rf = np.random.default_rng(seed); rf.shuffle(multi)
    folds = np.array_split(np.arange(len(multi)), int(round(1 / frac)))[:n_fold]; hid = []; fold_conv = []
    for fi, ix in enumerate(folds):
        hide = {multi[j] for j in ix}
        Df = reconcile_dc(G, model, {k: m for k, m in M.items() if k[1] not in hide}, dom=dom, n_starts=0, n_shuffle=1)
        Qpf, cf = passive_fit(Df, A); fold_conv.append(cf)
        hid += rows_for(Df["Qp"], Qpf, [(eix[e], m) for e in hide for m in by_e[e] if abs(m["Q"]) > 0], model)
        log(f"  fold {fi + 1}/{n_fold}: passive fit projected gradient {cf['rel_proj_grad']:.1e} ({cf['n_iter']} iterations)")
    hidden = paired(hid, rng)
    for lab, R in (("measured", measured), ("hidden", hidden)):
        for ref in ("trusted", "all"):
            r = R[ref]
            log(f"  {lab:8s} {ref:7s}: error {r['err_cons']:.3f} → {r['err_pass']:.3f} (Δ {r['d_err']:+.3f} [{r['d_err_ci'][0]:+.3f}, {r['d_err_ci'][1]:+.3f}]); "
                f"direction {100 * r['dir_cons']:.1f} → {100 * r['dir_pass']:.1f} % (Δ {100 * r['d_dir']:+.1f} [{100 * r['d_dir_ci'][0]:+.1f}, {100 * r['d_dir_ci'][1]:+.1f}])")
    log(f"  totals: {json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in totals.items()})}")
    out = dict(measured=measured, hidden=hidden, totals=totals, convergence=dict(full=conv, folds=fold_conv), trust_c=TRUST, n_fold=n_fold, frac=frac, seed=seed)
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "dc_passive_comparison.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {P.RESULTS / 'dc_passive_comparison.json'}")


if __name__ == "__main__":
    main()
