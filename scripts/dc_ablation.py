"""Ablation of the mean-flow reconstruction (input to Fig 6d): which components improve prediction of held-out recordings?

Factors (2 × 2 × 2):  heart-rate normalisation (off / on, plexusflow.reconcile.gain) × measurement weighting (constant relative
uncertainty / alignment-dependent, plexusflow.reconcile.reliability) × reconstruction (per-vessel weighted average of the
remaining recordings only / conservative reconciliation, plexusflow.reconcile.dc).
Validation: leave-one-recording-out (dc.repeat_folds, 5 folds); in every fold β, both uncertainty models and the reconstruction
are refitted without the held-out recordings.  Each held-out recording is predicted in ORIGINAL units (reference-scale estimate ×
the recording's own heart-rate gain) and scored by |ln(predicted / recorded)| and direction.  All eight variants are scored on
the SAME held-out recordings (the intersection: trusted, c < 0.5 under the production model, and predictable by every variant).
Uncertainty: tile bootstrap (held-out recordings resampled by recording tile), paired across variants.
Writes $PLEXUSFLOW_RESULTS/dc_ablation.json.  ~2 min.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import plexusflow.paths as P
from plexusflow.reconcile import measurement_table, carved_domain, reconcile_dc, fit_reliability
from plexusflow.reconcile.gain import tile_heart_rates, fit_hr_gain, apply_hr_gain
from plexusflow.reconcile.reliability import c_of
from plexusflow.reconcile.dc import repeat_folds, vessel_input
from dc_withholding_curve import fit_constant
T0 = time.time(); log = lambda s: print(f"[+{time.time() - T0:4.0f} s] {s}", flush=True)


def main(n_boot=2000, seed=0):
    G = P.load_graph(); f0 = tile_heart_rates(G); R0 = measurement_table(G, gain=None); dom = carved_domain(G)
    prod, _ = fit_reliability(G, measurement_table(G)); trusted = lambda m: c_of(prod, m) < 0.5
    pred = defaultdict(dict); obs = {}; tile_of = {}
    for k_fold, (Mf_raw, held) in enumerate(repeat_folds(R0, n_fold=5, seed=seed)):
        fit = fit_hr_gain(Mf_raw, f0)
        for hr in (False, True):
            Mf = apply_hr_gain(Mf_raw, f0, fit) if hr else Mf_raw
            gain = (lambda t: (f0[t] / fit["f_ref"]) ** fit["beta"]) if hr else (lambda t: 1.0)
            modA, _ = fit_reliability(G, Mf); mods = {"alignment": modA, "constant": fit_constant(G, Mf, modA)[0]}
            for wl, md in mods.items():
                D = reconcile_dc(G, md, Mf, dom=dom, n_starts=0, n_shuffle=0); ix = {tuple(e): i for i, e in enumerate(D["edges"])}
                recs = defaultdict(list)
                for (t, e), m in Mf.items(): recs[e].append((m, c_of(md, m)))
                for e, m in held:
                    if not trusted(m) or m["Q"] == 0: continue
                    key = (k_fold, m["tile"], tuple(e)); obs[key] = m["Q"]; tile_of[key] = m["tile"]
                    i = ix.get(tuple(e))
                    if i is not None: pred[(hr, wl, "conservation")][key] = D["Qp"][i] * gain(m["tile"])
                    if e in recs: pred[(hr, wl, "average")][key] = vessel_input(recs[e], "average")[0] * gain(m["tile"])
            log(f"fold {k_fold + 1}/5, HR {'on' if hr else 'off'}: β {fit['beta']:.2f}")
    keys = sorted(set.intersection(*[set(p) for p in pred.values()]))
    y = np.array([obs[k] for k in keys]); tl = np.array([tile_of[k] for k in keys]); ut = np.unique(tl); grp = [np.flatnonzero(tl == t) for t in ut]
    V = sorted(pred, key=lambda v: (v[2] == "conservation", v[1] == "alignment", v[0]))
    E = {v: np.abs(np.log(np.maximum(np.abs([pred[v][k] for k in keys]), 1e-12) / np.abs(y))) for v in V}
    S = {v: np.sign([pred[v][k] for k in keys]) == np.sign(y) for v in V}
    rng = np.random.default_rng(seed); boots = [np.concatenate([grp[j] for j in rng.integers(0, len(grp), len(grp))]) for _ in range(n_boot)]
    final = (True, "alignment", "conservation"); out = dict(n_heldout=len(keys), n_tiles=len(ut), variants=[])
    for v in V:
        med = np.median(E[v]); bs = np.array([np.median(E[v][b]) for b in boots]); dd = np.array([np.median(E[v][b]) - np.median(E[final][b]) for b in boots])
        row = dict(hr=v[0], weighting=v[1], reconstruction=v[2], fold_error=float(np.exp(med)), fold_error_ci=np.exp(np.percentile(bs, [2.5, 97.5])).tolist(),
                   direction=float(S[v].mean()), d_log_err_vs_final=float(med - np.median(E[final])), d_ci=np.percentile(dd, [2.5, 97.5]).tolist())
        out["variants"].append(row)
        log(f"HR {'on ' if v[0] else 'off'} | {v[1]:9s} | {v[2]:12s}: ×{row['fold_error']:.3f} [{row['fold_error_ci'][0]:.3f}, {row['fold_error_ci'][1]:.3f}], "
            f"direction {row['direction']:.1%}, Δ vs final {row['d_log_err_vs_final']:+.3f} [{row['d_ci'][0]:+.3f}, {row['d_ci'][1]:+.3f}]")
    log(f"{len(keys)} held-out recordings common to all variants ({len(ut)} tiles)")
    P.RESULTS.mkdir(parents=True, exist_ok=True); (P.RESULTS / "dc_ablation.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {P.RESULTS / 'dc_ablation.json'}")


if __name__ == "__main__":
    main()
