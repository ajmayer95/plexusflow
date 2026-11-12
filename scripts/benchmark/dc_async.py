"""Known-truth benchmark of the DC reconstruction when the tiles see DIFFERENT flow fields (asynchronous acquisition of a
changing network) or carry tile-level velocity-scale bias — reviewer request 2026-11-04.  Real 21S network, measurement
mask, reliability c_k, floor s0 and each measured vessel's recording tile (as scripts/benchmark/dc_reconstruction.py).

Per realisation and tile α: boundary fluxes f̄ × exp(τ_local N_n) (per boundary node) × exp(τ_global N_α) (per tile), veins
rebalanced → passive field Q*(t_α) (fixed geometry).  A per-tile global factor is equivalent to a per-tile velocity-scale
bias (y = g_α Q): both multiply every flow of a tile and leave its own junctions balanced.  Vessel k is measured in its
winner tile: y_k = Q*_k(t_α(k)) + √(c²Q*² + s0²) N(0,1);  repeat measurements of the real repeat vessels are simulated in
their other tiles (for matching to the real repeat disagreement).  Reference truth: the time-average Q̄ = mean_α Q*(t_α)
(conservative).  Reconstruction: production conservation-only quasi-likelihood.
Reported per (τ_local, τ_global): departure of the measured fields from Q̄ (median |log(Q*(t_α)/Q̄)| over measured vessels);
reconstruction error and direction vs Q̄ (measured / hidden-determined / hidden-on-loops); the 2σ mask (|Q′| > 2σ,
σ² = c²Q′² + s0²): its size and the fraction of its vessels with the right direction; same-tile junction RMS z under the c
model (does the calibration see the error?); synthetic repeat disagreement median |log(y/y_rep)| (trusted c < 0.5 / all),
to compare with the real data (scripts/dc_validate_weighting.py, 'raw').
Writes $PLEXUSFLOW_RESULTS/dc_async_benchmark.json.   Usage: python scripts/benchmark/dc_async.py [--reps 20] [--jobs 8]
"""
from __future__ import annotations
import argparse, json, sys, time
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.reconcile.network import project, passive, EPS_UNIMAGED
from plexusflow.reconcile.dc import ql_solve
from plexusflow.reconcile.reliability import measurement_table, c_of

TAU_LOCAL = (0.0, 0.1, 0.2, 0.3, 0.5)
TAU_GLOBAL = (0.0, 0.1, 0.2, 0.4)
TRUST = 0.5


@lru_cache(maxsize=1)
def ctx():
    import dc_reconstruction as base
    X = dict(base.ctx()); dom = X["dom"]; im = X["im"]
    import pickle
    D = pickle.loads((paths.RESULTS / "dc_reconciled_v2.pkl").read_bytes()); model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text())
    G = paths.load_graph(); M = measurement_table(G); eix = {tuple(e): i for i, e in enumerate(dom["edges"])}
    X["mtile"] = np.asarray(D["tile"]); tiles = sorted(set(int(t) for t in X["mtile"][im]))
    X["tiles"] = tiles; X["tix"] = np.array([tiles.index(int(t)) if im[i] else -1 for i, t in enumerate(X["mtile"])])
    rep = [(eix[e], tiles.index(m["tile"]) if m["tile"] in tiles else -1, c_of(model, m)) for (t, e), m in M.items()
           if e in eix and im[eix[e]] and m["tile"] != X["mtile"][eix[e]]]
    X["rep"] = [(i, a, c) for i, a, c in rep if a >= 0]
    B = dom["B"].tocsr(); J = []                                               # same-tile junctions (all incident vessels' winners in one tile)
    for n in np.setdiff1d(np.arange(B.shape[0]), dom["bnd"]):
        es, sg = B[n].indices, B[n].data
        if len(es) >= 3 and im[es].all() and len(set(X["tix"][es])) == 1:
            J.append((es, sg))
    X["junctions"] = J
    return X


def realisation(tl, tg, rep_i):
    X = ctx(); dom, a, l, im, C, s0, loop = X["dom"], X["a"], X["l"], X["im"], X["C"], X["s0"], X["loop"]
    rng = np.random.default_rng(10007 * rep_i + int(1000 * tl) + int(100 * tg)); K = a ** 4 / l
    inj, out = np.asarray(dom["inj"]), np.asarray(dom["out"]); nT = len(X["tiles"])
    loc = np.exp(tl * rng.standard_normal(len(inj))); fields = []
    base_loc = loc                                                            # one realisation-wide redistribution baseline …
    for t in range(nT):
        fb = X["fb"].copy(); fb[inj] *= base_loc * np.exp(tl * rng.standard_normal(len(inj))) * np.exp(tg * rng.standard_normal())
        fb[out] *= -fb[inj].sum() / fb[out].sum(); fields.append(passive(K, fb, dom))   # … plus per-tile redistribution and scale
    F = np.array(fields); Qbar = F.mean(0)
    Qmeas = np.where(im, F[np.maximum(X["tix"], 0), np.arange(F.shape[1])], 0.0)
    y = np.where(im, Qmeas + np.sqrt(C ** 2 * Qmeas ** 2 + s0 ** 2) * rng.standard_normal(len(Qbar)), 0.0)
    Q0 = project(y, np.where(im, 1.0, EPS_UNIMAGED), dom); m = float(np.median(1 / (C[im] ** 2 * Q0[im] ** 2 + s0 ** 2)))
    Qp, _ = ql_solve(Q0, y, C, im, s0, m, dom)
    sig = np.sqrt(C ** 2 * Qp ** 2 + s0 ** 2)
    def score(sel):
        aq = np.abs(Qbar); v = sel & (aq >= np.percentile(aq[sel], 10))
        return float(np.median(np.abs(np.log(np.maximum(np.abs(Qp[v]), 1e-12) / aq[v])))), float(np.mean(np.sign(Qp[v]) == np.sign(Qbar[v])))
    out_ = dict(tau_local=tl, tau_global=tg, rep=rep_i)
    v = im & (np.abs(Qbar) >= np.percentile(np.abs(Qbar[im]), 10))
    out_["departure"] = float(np.median(np.abs(np.log(np.abs(Qmeas[v]) / np.abs(Qbar[v])))))
    for lab, sel in (("measured", im), ("hidden_det", ~im & ~loop), ("hidden_loop", loop)):
        out_[f"err_{lab}"], out_[f"dir_{lab}"] = score(sel)
    mask = im & (np.abs(Qp) > 2 * sig); out_["mask_frac"] = float(mask.sum() / im.sum())
    out_["mask_dir"] = float(np.mean(np.sign(Qp[mask]) == np.sign(Qbar[mask])))
    out_["mask_dir_inst"] = float(np.mean(np.sign(Qp[mask]) == np.sign(Qmeas[mask])))      # vs the field its tile actually saw
    z = []
    for es, sg in X["junctions"]:
        z.append((sg * y[es]).sum() / np.sqrt(np.sum((C[es] * y[es]) ** 2) + s0 ** 2))
    out_["junction_rms_z"] = float(np.sqrt(np.mean(np.square(z))))
    ri = np.array([r[0] for r in X["rep"]]); ra = np.array([r[1] for r in X["rep"]]); rc = np.array([r[2] for r in X["rep"]])
    qr = F[ra, ri]; yr = qr + np.sqrt(rc ** 2 * qr ** 2 + s0 ** 2) * rng.standard_normal(len(qr)); ok = (np.abs(yr) > 0) & (np.abs(y[ri]) > 0)
    d = np.abs(np.log(np.abs(y[ri][ok]) / np.abs(yr[ok]))); tr = rc[ok] < TRUST
    out_["repeat_trusted"], out_["repeat_all"] = float(np.median(d[tr])), float(np.median(d))
    out_["repeat_dir_trusted"] = float(np.mean(np.sign(y[ri][ok][tr]) == np.sign(yr[ok][tr])))
    return out_


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); t0 = time.time(); X = ctx()
    print(f"{paths.STAGE.name}: {len(X['tiles'])} tiles, {X['im'].sum()} measured vessels, {len(X['rep'])} repeat measurements, "
          f"{len(X['junctions'])} same-tile junctions", flush=True)
    import dc_async as _mod
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_mod.realisation)(tl, tg, r) for tl in TAU_LOCAL for tg in TAU_GLOBAL for r in range(a.reps))
    keys = [k for k in rows[0] if k not in ("tau_local", "tau_global", "rep")]
    table = []
    for tl in TAU_LOCAL:
        for tg in TAU_GLOBAL:
            rs = [r for r in rows if r["tau_local"] == tl and r["tau_global"] == tg]
            table.append(dict(tau_local=tl, tau_global=tg, **{k: float(np.median([r[k] for r in rs])) for k in keys},
                              **{k + "_iqr": [float(np.percentile([r[k] for r in rs], q)) for q in (25, 75)] for k in keys}))
    print("\n τloc τglob | departure | err meas / hid-det / hid-loop | dir meas / hid-det | 2σ mask: frac, dir vs Q̄ (vs own tile) | junction RMS z | repeat trusted / all (real 0.345 / 0.664)")
    for t in table:
        print(f"  {t['tau_local']:.1f}  {t['tau_global']:.1f}  |   {t['departure']:.3f}   | {t['err_measured']:.3f} / {t['err_hidden_det']:.3f} / {t['err_hidden_loop']:.3f} | "
              f"{100 * t['dir_measured']:.1f} / {100 * t['dir_hidden_det']:.1f} | {t['mask_frac']:.2f}, {100 * t['mask_dir']:.1f} % ({100 * t['mask_dir_inst']:.1f}) | "
              f"{t['junction_rms_z']:.2f} | {t['repeat_trusted']:.3f} / {t['repeat_all']:.3f}")
    out = dict(stage=paths.STAGE.name, tau_local=TAU_LOCAL, tau_global=TAU_GLOBAL, reps=a.reps, rows=rows, table=table,
               real_repeat=dict(trusted=0.345, all=0.664, source="dc_weighting_validation.json measured.raw (21S)"))
    paths.RESULTS.mkdir(parents=True, exist_ok=True); (paths.RESULTS / "dc_async_benchmark.json").write_text(json.dumps(out))
    print(f"\nwrote {paths.RESULTS / 'dc_async_benchmark.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
