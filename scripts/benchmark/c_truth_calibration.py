"""Calibration of the reliability scale c against known truth (input to SI fig6_S_reliability).

Reads the Stokes junction ensemble (scripts/benchmark/junctions.py output: <results>/stokes_bench/ensemble/*.pkl — synthetic
videos of 27 real junctions with exactly known, conserving flow, analysed by the unchanged production pipeline) and scores
each measured edge with the CURRENT reliability model (results/reliability_model_v1.json).  Error = log(Q̂/Q_true) with the
common-mode under-read (median, ×0.71; invisible to c and to junction balance) removed.  Per c-bin: mean c, RMS and robust SD
of the error, 95 % intervals from a bootstrap over junctions (edges of one junction share a video).  Also reports the
Kloosterman-failure and sign-error rates and the Spearman rank correlation of |error| with c.
Writes $PLEXUSFLOW_RESULTS/c_truth_calibration.json.
"""
from __future__ import annotations
import glob, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from scipy.stats import spearmanr
import plexusflow.paths as paths

BINS = (0.0, 0.2, 0.3, 0.45, 0.7, 1.0, 9.0)


def main():
    m = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text()); c0 = m.get("c0", np.median(list(m["c_T"].values())))
    term = m.get("profile_term", "f"); coef = m[f"c_{term}"]                       # profile term of the reliability model (p or f)
    rows = []
    for f in sorted(glob.glob(str(paths.RESULTS / "stokes_bench" / "ensemble" / "*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for _, cs in d["cases"].items():
            for e in cs["edges"]:
                al = e.get("align", np.nan); al = al if np.isfinite(al) else 1.0
                rows.append((e.get(term, np.nan), al, e["Q_true"], e["Q_kl"], int(d["node"])))
    A = np.array(rows, float); f, al, Qt, Qm, node = A.T
    measured = np.isfinite(Qm) & (f > 0); ok = measured & (np.sign(Qm) == np.sign(Qt)) & (Qt != 0)
    c = c0 + coef * (1 - f) + m["c_a"] * (1 - al)
    e = np.log(np.abs(Qm[ok]) / np.abs(Qt[ok])); common = float(np.median(e)); e = e - common; cc = c[ok]; nd = node[ok]
    rsd = lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))
    rng = np.random.default_rng(0); juncs = np.unique(nd); bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (cc >= lo) & (cc < hi)
        if s.sum() < 20:
            continue
        bs_rms, bs_c = [], []
        for _ in range(1000):
            pick = rng.choice(juncs, len(juncs)); w = np.concatenate([np.flatnonzero(s & (nd == j)) for j in pick])
            if len(w) > 5:
                bs_rms.append(np.sqrt(np.mean(e[w] ** 2))); bs_c.append(cc[w].mean())
        bins.append(dict(lo=lo, hi=hi, n=int(s.sum()), c_mean=float(cc[s].mean()), rms=float(np.sqrt(np.mean(e[s] ** 2))), robust_sd=rsd(e[s]),
                         rms_ci=np.percentile(bs_rms, [2.5, 97.5]).tolist()))
    sign = measured & (np.sign(Qm) != np.sign(Qt))
    out = dict(n_edges=int(len(A)), n_usable=int(ok.sum()), n_junctions=int(len(juncs)), common_mode=float(np.exp(common)),
               failure_rate=float(np.mean(~measured)), sign_error_rate=float(np.mean(sign)), c_median_sign_errors=float(np.median(c[sign])),
               c_median_good=float(np.median(cc)), spearman=float(spearmanr(np.abs(e), cc)[0]), bins=bins, model=m)
    for b in bins:
        print(f"c {b['lo']:.2f}–{b['hi']:.2f}: n {b['n']:4d}  mean c {b['c_mean']:.2f}  RMS error {b['rms']:.3f} [{b['rms_ci'][0]:.3f}, {b['rms_ci'][1]:.3f}]  robust SD {b['robust_sd']:.3f}")
    print(f"{out['n_usable']} of {out['n_edges']} edges, {out['n_junctions']} junctions; common mode ×{out['common_mode']:.2f}; Spearman {out['spearman']:+.2f}; "
          f"failures {100 * out['failure_rate']:.1f} %, sign errors {100 * out['sign_error_rate']:.1f} % (median c {out['c_median_sign_errors']:.2f})")
    (paths.RESULTS / "c_truth_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
