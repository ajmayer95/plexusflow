"""Validation of the DC reconciliation weighting against independent repeat recordings (input to Fig 6f).

Carved vessels measured in ≥ 2 tiles provide independent references.  With the production input (per_vessel="average": all
recordings of a vessel averaged) one recording per repeat-measured vessel is held out per fold (dc.repeat_folds, 5 folds);
with winner input, every non-winner recording is held out.  Two tests, five weightings:
  measured   the reconciled flow of a vessel vs its held-out repeat (does imposing conservation degrade the data?)
  imputed    hide EVERY measurement of a random 10 % of the repeat-measured vessels (5 disjoint folds), reconcile the
             rest, compare the network-imputed flow with the hidden measurements (does conservation add information?)
Weightings: uniform w = 1;  quality w = 1/c²;  measured-scale w = 1/(c² y² + s0²);  calibrated (production)
variance c² Q′² + s0² at the reconciled flow (reconcile_dc quasi-likelihood; no perturbed starts in the imputation folds);  and 'raw'
(Q = y, measured test only).  Scores: median |log(|Q|/|h|)| and direction agreement, for all references and for
trusted references (c_h < 0.5); 95 % intervals from a vessel-clustered bootstrap.
Writes $PLEXUSFLOW_RESULTS/dc_weighting_validation.json.  ~5 min (21S).
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plexusflow.paths as P
from plexusflow.reconcile.reliability import measurement_table, c_of
from plexusflow.reconcile.dc import reconcile_dc, repeat_folds
from plexusflow.reconcile.network import project, carved_domain, EPS_UNIMAGED

KINDS = ("uniform", "quality", "measured_scale", "calibrated")
TRUST = 0.5
T0 = time.time()
log = lambda s: print(f"[+{time.time() - T0:4.0f} s] {s}", flush=True)


def fields(D, dom):
    y, im = D["Qm"], D["imaged"]; C = np.where(im, D["c"], 1.0); s0 = D["sigma0_floor"]
    norm = lambda w: np.where(im, w / np.median(w[im]), EPS_UNIMAGED)
    return {"uniform": project(y, norm(np.ones_like(y)), dom), "quality": project(y, norm(1 / C ** 2), dom),
            "measured_scale": project(y, norm(1 / (C ** 2 * y ** 2 + s0 ** 2)), dom), "calibrated": D["Qp"]}


def paired(rows, rng, ref="calibrated", n_boot=1000):
    """Paired differences (alternative − reference method) on identical references: median error and direction rate, with
    95 % intervals from a bootstrap over vessels and over reference tiles (recordings in a tile share errors / physiology)."""
    out = {}
    A = {k: np.array(v, float) for k, v in rows.items()}
    base = A[ref]
    for k, a in A.items():
        if k == ref:
            continue
        assert np.array_equal(a[:, 0], base[:, 0]) and np.array_equal(a[:, 4], base[:, 4])
        res = {}
        for lab, sel in (("trusted", base[:, 3] < TRUST), ("all", np.ones(len(base), bool))):
            b0, b1 = base[sel], a[sel]
            d_err = float(np.median(b1[:, 1]) - np.median(b0[:, 1])); d_dir = float(b1[:, 2].mean() - b0[:, 2].mean())
            ci = {}
            for unit, col in (("vessel", 0), ("tile", 4)):
                ids = np.unique(b0[:, col]); grp = [np.nonzero(b0[:, col] == v)[0] for v in ids]; de, dd = [], []
                for _ in range(n_boot):
                    ix = np.concatenate([grp[j] for j in rng.integers(0, len(grp), len(grp))])
                    de.append(np.median(b1[ix, 1]) - np.median(b0[ix, 1])); dd.append(b1[ix, 2].mean() - b0[ix, 2].mean())
                ci[unit] = dict(d_err=np.percentile(de, [2.5, 97.5]).tolist(), d_dir=np.percentile(dd, [2.5, 97.5]).tolist(), n_units=int(len(ids)))
            res[lab] = dict(d_err=d_err, d_dir=d_dir, ci=ci, n=int(sel.sum()))
        out[k] = res
    return out


def summarise(rows, rng, n_boot=500):
    """rows: (vessel, err, direction_ok, c_ref) → {all, trusted}: median err [CI], direction agreement [CI], n"""
    a = np.array(rows, float); out = {}
    for lab, sel in (("all", np.ones(len(a), bool)), ("trusted", a[:, 3] < TRUST)):
        b = a[sel]; ves = np.unique(b[:, 0]); grp = [np.nonzero(b[:, 0] == v)[0] for v in ves]; be, bd = [], []
        for _ in range(n_boot):
            ix = np.concatenate([grp[k] for k in rng.integers(0, len(grp), len(grp))]); be.append(np.median(b[ix, 1])); bd.append(b[ix, 2].mean())
        out[lab] = dict(err=float(np.median(b[:, 1])), err_ci=np.percentile(be, [2.5, 97.5]).tolist(),
                        dir=float(b[:, 2].mean()), dir_ci=np.percentile(bd, [2.5, 97.5]).tolist(), n=int(len(b)), n_vessels=int(len(ves)))
    return out


def main(n_fold=5, frac=0.10, seed=0):
    G = P.load_graph(); M = measurement_table(G); model = json.loads((P.RESULTS / "reliability_model_v1.json").read_text())
    D = pickle.loads((P.RESULTS / "dc_reconciled_v2.pkl").read_bytes()); dom = carved_domain(G)
    E = list(dom["edges"]); assert E == [tuple(e) for e in D["edges"]]; eix = {e: i for i, e in enumerate(E)}
    by_e = {}
    for (t, e), m in M.items():
        if e in eix:
            by_e.setdefault(e, []).append(m)
    rng = np.random.default_rng(seed); rows = {k: [] for k in ("raw",) + KINDS}
    # 1. measured vessels: reconciled vs held-out repeat
    def add(F, refs):
        for i, m in refs:
            if abs(m["Q"]) > 0:
                for k, Q in F.items():
                    rows[k].append((i, abs(np.log(max(abs(Q[i]), 1e-12) / abs(m["Q"]))), float(np.sign(Q[i]) == np.sign(m["Q"])), c_of(model, m), int(m["tile"])))
    if D.get("per_vessel", "winner") == "average":           # every recording is input: hold one out per repeat-measured vessel
        for fi, (Mf, held) in enumerate(repeat_folds(M, n_fold, seed)):
            Df = reconcile_dc(G, model, Mf, dom=dom, n_starts=0, n_shuffle=1)
            add({"raw": Df["Qm"], **fields(Df, dom)}, [(eix[e], m) for e, m in held if e in eix])
            log(f"measured, leave-one-recording-out fold {fi + 1}/{n_fold}: {len(held)} recordings held out")
    else:                                                    # winner input: every other recording is independent
        add({"raw": D["Qm"], **fields(D, dom)}, [(eix[e], m) for e, ms in by_e.items() for m in ms if m["tile"] != D["tile"][eix[e]]])
    measured = {k: summarise(v, rng) for k, v in rows.items()}; measured_paired = paired(rows, np.random.default_rng(seed + 101))
    log(f"measured: {measured['calibrated']['all']['n_vessels']} vessels with repeats; trusted err raw {measured['raw']['trusted']['err']:.3f}, "
        f"calibrated {measured['calibrated']['trusted']['err']:.3f}")
    # 2. imputation: hide every measurement of 10 % of the repeat-measured vessels per fold
    multi = sorted(e for e, ms in by_e.items() if len(ms) >= 2); rng.shuffle(multi)
    folds = np.array_split(np.arange(len(multi)), int(round(1 / frac)))[:n_fold]; rows = {k: [] for k in KINDS}
    for fi, ix in enumerate(folds):
        hide = {multi[j] for j in ix}
        Df = reconcile_dc(G, model, {k: m for k, m in M.items() if k[1] not in hide}, dom=dom, n_starts=0, n_shuffle=1); Ff = fields(Df, dom)
        for e in hide:
            i = eix[e]
            for m in by_e[e]:
                if abs(m["Q"]) > 0:
                    for k, Q in Ff.items():
                        rows[k].append((i, abs(np.log(max(abs(Q[i]), 1e-12) / abs(m["Q"]))), float(np.sign(Q[i]) == np.sign(m["Q"])), c_of(model, m), int(m["tile"])))
        log(f"imputation fold {fi + 1}/{n_fold}: {len(hide)} vessels hidden")
    imputed = {k: summarise(v, rng) for k, v in rows.items()}; imputed_paired = paired(rows, np.random.default_rng(seed + 102))
    for lab, PP in (("reconciliation", measured_paired), ("imputation", imputed_paired)):
        for k, r in PP.items():
            t = r["trusted"]
            log(f"  paired {lab:14s} {k:15s} − calibrated (trusted): Δerr {t['d_err']:+.3f} vessel CI [{t['ci']['vessel']['d_err'][0]:+.3f}, "
                f"{t['ci']['vessel']['d_err'][1]:+.3f}] tile CI [{t['ci']['tile']['d_err'][0]:+.3f}, {t['ci']['tile']['d_err'][1]:+.3f}] "
                f"({t['ci']['tile']['n_units']} tiles); Δdir {100 * t['d_dir']:+.1f} pts tile CI [{100 * t['ci']['tile']['d_dir'][0]:+.1f}, {100 * t['ci']['tile']['d_dir'][1]:+.1f}]")
    for k in KINDS:
        log(f"  {k:15s} measured trusted {measured[k]['trusted']['err']:.3f} / {measured[k]['trusted']['dir']:.3f}   "
            f"imputed trusted {imputed[k]['trusted']['err']:.3f} / {imputed[k]['trusted']['dir']:.3f}")
    out = dict(measured=measured, imputed=imputed, measured_paired=measured_paired, imputed_paired=imputed_paired, trust_c=TRUST, n_fold=n_fold, frac=frac, seed=seed)
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "dc_weighting_validation.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {P.RESULTS / 'dc_weighting_validation.json'}")


if __name__ == "__main__":
    main()
