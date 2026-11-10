"""Junction calibration of the measurement-discrepancy scale: quality-dependent c_k vs one constant c (input to Fig 6b).

Same-tile junctions (every incident vessel Kloosterman-valid in one recording) give r_j = Σ s_k y_k with
    E[r_j²] ≈ Σ c_k² y_k² + σ0²,    c_k = c0 + c_p (1 − p_k) + c_a (1 − align_k)     (production, fit_reliability; p = parabolicity)
or with one constant c.  Both models are refitted on the junctions of 4/5 of the tiles (whole tiles per fold, 5 folds, the same
tile split as make_figure6.heldout_calibration) and predict σ_j for the held-out tiles.  Reports, per quintile of junction
quality (mean production c_k of its vessels): RMS of the out-of-fold standardized imbalance z_j = r_j / σ_j for both models,
with 95 % tile-bootstrap intervals; held-out NLL of both; and the Kloosterman-style relative imbalance δ = (in − out)/in
(robust SD, mean, single fractional error SD(δ/√(Σy²/in²))).  c is an effective discrepancy scale (bias + noise), not a
noise SD.  Also the same comparison in physical units: observed RMS of |r_j| / √Σ y_k² per quintile vs the scale each
model predicts (Fig 6b).  Writes $PLEXUSFLOW_RESULTS/dc_junction_calibration.json.  ~1 min.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import plexusflow.paths as P
from plexusflow.reconcile.reliability import measurement_table, same_tile_junctions, _arrays, _fit_theta, _nll, profile_coef

N_FOLD, N_Q, N_BOOT, SEED = 5, 5, 1000, 0
T0 = time.time()
log = lambda s: print(f"[+{time.time() - T0:4.0f} s] {s}", flush=True)
madsd = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))


def predict(d, th, sel):
    """σ_j for junctions `sel` under θ = (log c0, log c_f, log c_a, log σ0) — production _c with one tile term."""
    idx, dc, f, al, _, net = d; c = np.exp(th[0]) + np.exp(th[1]) * (1 - f) + np.exp(th[2]) * (1 - al)
    return np.sqrt(np.bincount(idx, (dc * c) ** 2, minlength=len(net)) + np.exp(2 * th[3]))[sel]


def subset(d, sel):
    idx, dc, f, al, ti, net = d; keep = sel[idx]; remap = np.cumsum(sel) - 1
    return remap[idx[keep]], dc[keep], f[keep], al[keep], ti[keep], net[sel]


def main():
    G = P.load_graph(); M = measurement_table(G); model = json.loads((P.RESULTS / "reliability_model_v1.json").read_text())
    tiles = sorted({t for t, _ in M}); J = []
    for t in tiles:
        J += [(t, j) for j in same_tile_junctions(G, t, {e for (tt, e) in M if tt == t})]
    term, coef = profile_coef(model)                                                           # profile term: p (or f in old models)
    d = _arrays(J, M, tiles, term); d = d[:4] + (np.zeros_like(d[4]),) + d[5:]                      # one tile term (production form)
    dconst = (d[0], d[1], np.ones_like(d[2]), np.ones_like(d[3]), d[4], d[5])                 # f = align = 1 → c = c0 constant
    idx, dc, f, al, _, net = d; jt = np.array([t for t, _ in J]); nJ = len(J)
    log(f"{P.STAGE.name}: {nJ} same-tile junctions in {len(tiles)} tiles")
    # out-of-fold σ_j (tile folds identical to make_figure6.heldout_calibration)
    fold = dict(zip(tiles, np.random.default_rng(SEED).permutation(len(tiles)) % N_FOLD)); jf = np.array([fold[t] for t in jt])
    sig = {"calibrated": np.empty(nJ), "constant": np.empty(nJ)}; nll = {"calibrated": 0.0, "constant": 0.0}
    for k in range(N_FOLD):
        tr, te = jf != k, jf == k
        for lab, dd in (("calibrated", d), ("constant", dconst)):
            th = _fit_theta(subset(dd, tr), 1).x
            sig[lab][te] = predict(dd, th, te); nll[lab] += float(_nll(th, subset(dd, te), 1, grad=False))
            if lab == "constant":
                log(f"  fold {k}: constant c {np.exp(th[0]):.3f}, σ0 {np.exp(th[3]):.1e}; {te.sum()} held-out junctions")
    # junction quality = mean production c_k of its vessels (fixed, from the stored model)
    ck = model["c0"] + coef * (1 - f) + model["c_a"] * (1 - al); jq = np.bincount(idx, ck, minlength=nJ) / np.bincount(idx, minlength=nJ)
    qe = np.quantile(jq, np.linspace(0, 1, N_Q + 1)); qb = np.clip(np.searchsorted(qe, jq, side="right") - 1, 0, N_Q - 1)
    ti = np.searchsorted(tiles, jt); rng = np.random.default_rng(SEED + 1); out = dict(quintile_edges=qe.tolist(),
                                                                                    quintile_median_c=[float(np.median(jq[qb == b])) for b in range(N_Q)])
    for lab in sig:
        z2 = (net / sig[lab]) ** 2; S = np.zeros((len(tiles), N_Q)); N = np.zeros((len(tiles), N_Q))
        np.add.at(S, (ti, qb), z2); np.add.at(N, (ti, qb), 1)
        boot = np.array([np.sqrt(S[p].sum(0) / np.maximum(N[p].sum(0), 1)) for p in rng.integers(0, len(tiles), (N_BOOT, len(tiles)))])
        rms = np.sqrt(S.sum(0) / N.sum(0)); ci = np.percentile(boot, [2.5, 97.5], axis=0)
        cov = [float(np.mean(np.abs(net[qb == b]) / sig[lab][qb == b] < 1)) for b in range(N_Q)]
        out[lab] = dict(rms_z=rms.tolist(), rms_z_ci=ci.T.tolist(), coverage_1sd=cov, rms_z_all=float(np.sqrt(np.mean(z2))),
                        heldout_nll=nll[lab], n=N.sum(0).astype(int).tolist())
        log(f"  {lab:10s}: out-of-fold RMS z by quality quintile " + " ".join(f"{x:.2f}" for x in rms) + f"; overall {np.sqrt(np.mean(z2)):.2f}; "
            f"held-out NLL {nll[lab]:.1f}")
    log(f"  held-out NLL constant − calibrated: {nll['constant'] - nll['calibrated']:+.1f}")
    # the same comparison in physical terms: observed relative imbalance |r_j| / √Σ y_k² vs the scale each model predicts for it
    S = np.sqrt(np.bincount(idx, dc ** 2, minlength=nJ)); rel2 = (net / S) ** 2
    So = np.zeros((len(tiles), N_Q)); No = np.zeros((len(tiles), N_Q)); np.add.at(So, (ti, qb), rel2); np.add.at(No, (ti, qb), 1)
    boot = np.array([np.sqrt(So[p].sum(0) / np.maximum(No[p].sum(0), 1)) for p in rng.integers(0, len(tiles), (N_BOOT, len(tiles)))])
    out["relative"] = dict(observed=np.sqrt(So.sum(0) / No.sum(0)).tolist(), observed_ci=np.percentile(boot, [2.5, 97.5], axis=0).T.tolist(),
                           **{f"predicted_{lab}": [float(np.sqrt(np.mean((sig[lab][qb == b] / S[qb == b]) ** 2))) for b in range(N_Q)] for lab in sig})
    R_ = out["relative"]
    log("  relative imbalance RMS by quintile: observed " + " ".join(f"{x:.3f}" for x in R_["observed"]) + " | calibrated " +
        " ".join(f"{x:.3f}" for x in R_["predicted_calibrated"]) + " | constant " + " ".join(f"{x:.3f}" for x in R_["predicted_constant"]))
    # observed vs predicted (Fig 6b): junctions binned by their held-out PREDICTED relative imbalance σ_j / √Σ y_k²; per bin the RMS
    # predicted and the RMS observed |r_j| / √Σ y_k² (tile-bootstrap 95 %); the diagonal is perfect calibration
    N_B = 8; out["by_predicted"] = {}
    for lab in sig:
        pr = sig[lab] / S; be = np.quantile(pr, np.linspace(0, 1, N_B + 1)); bb = np.clip(np.searchsorted(be, pr, side="right") - 1, 0, N_B - 1)
        So_ = np.zeros((len(tiles), N_B)); No_ = np.zeros((len(tiles), N_B)); np.add.at(So_, (ti, bb), rel2); np.add.at(No_, (ti, bb), 1)
        bo = np.array([np.sqrt(So_[p_].sum(0) / np.maximum(No_[p_].sum(0), 1)) for p_ in rng.integers(0, len(tiles), (N_BOOT, len(tiles)))])
        out["by_predicted"][lab] = dict(predicted=[float(np.sqrt(np.mean(pr[bb == b] ** 2))) for b in range(N_B)],
                                        observed=np.sqrt(So_.sum(0) / No_.sum(0)).tolist(), observed_ci=np.percentile(bo, [2.5, 97.5], axis=0).T.tolist(),
                                        n=No_.sum(0).astype(int).tolist())
        B_ = out["by_predicted"][lab]
        log(f"  observed vs predicted ({lab}): " + "  ".join(f"{p_:.2f}→{o_:.2f}" for p_, o_ in zip(B_["predicted"], B_["observed"])))
    # Kloosterman-style relative imbalance
    yin = np.array([sum(max(-s * M[(t, e)]["Q"], 0) for e, s in j) for t, j in J]); yout = np.array([sum(max(s * M[(t, e)]["Q"], 0) for e, s in j) for t, j in J])
    ok = yin > 0; delta = (yin - yout)[ok] / yin[ok]; ratio = np.bincount(idx, dc ** 2, minlength=nJ)[ok] / yin[ok] ** 2
    out["kloosterman"] = dict(n=int(ok.sum()), delta_robust_sd=madsd(delta), delta_mean_trim=float(delta[np.abs(delta) < 1].mean()),
                              delta_median=float(np.median(delta)), single_fractional_error=madsd(delta / np.sqrt(ratio)),
                              hist_edges=np.linspace(-1, 1, 41).tolist(), hist=np.histogram(delta, np.linspace(-1, 1, 41))[0].tolist())
    log(f"  δ = (in − out)/in: robust SD {out['kloosterman']['delta_robust_sd']:.3f}, single fractional error {out['kloosterman']['single_fractional_error']:.3f}")
    out.update(n_junctions=nJ, n_tiles=len(tiles), n_fold=N_FOLD, seed=SEED, model=model)
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "dc_junction_calibration.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {P.RESULTS / 'dc_junction_calibration.json'}")


if __name__ == "__main__":
    main()
