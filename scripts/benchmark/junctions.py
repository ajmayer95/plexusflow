"""Known-truth junction ensemble: real Y-junctions × flow split α × speed × seed → production measurement vs exact truth.

Per edge: e = log(Q̂/Q_true) on sign-consistent cases (sign errors and profile-fit failures counted separately).
Prints: bias and scatter by speed, size, kept fraction f, alignment and L/R; whether the real reliability scale c
(results/reliability_model_v1.json, tile term = median c_T) tracks the true error; a fitted bias model; junction imbalance.
Writes <out>/junctions/<node>_a<α>_s<seed>.pkl (resumable) and <out>/junctions_summary.png.
Usage: python scripts/benchmark/junctions.py --bundle PATH [--out DIR] [--jobs 8] [--junctions N] [--summary-only]
"""
import sys, argparse, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from joblib import Parallel, delayed

import plexusflow.paths as paths
from plexusflow.synthetic.stokes.bundle import Bundle
from plexusflow.synthetic.stokes.cases import junction_case, JUNCTION_ALPHAS, JUNCTION_SEEDS


def run_one(bundle_root, out, n, alpha, seed):
    f = out / f"{n}_a{alpha:.1f}_s{seed}.pkl"
    if not f.exists():
        f.write_bytes(pickle.dumps(junction_case(Bundle(bundle_root), n, alpha, seed)))


def summarize(files, model_path, fig_path):
    from scipy.stats import spearmanr
    from scipy.optimize import minimize
    model = json.loads(Path(model_path).read_text()); cT0 = float(np.median(list(model["c_T"].values())))
    rows, imb = [], []
    for f in files:
        d = pickle.loads(Path(f).read_bytes())
        for v, c in d["cases"].items():
            imb.append((v, c["imbalance"]))
            for e in c["edges"]:
                al = e.get("align", np.nan); al = al if np.isfinite(al) else 1.0
                rows.append(dict(vpar=v, Rseg=e["R_seg"], v=e["v_true"], LR=e["L_over_R"], f=e["f"], align=al, Qt=e["Q_true"],
                                 Qm=e["Q_profile"], c=cT0 + model["c_" + model.get("profile_term", "f")] * (1 - e.get(model.get("profile_term", "f"), np.nan)) + model["c_a"] * (1 - al)))
    A = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    fail = ~np.isfinite(A["Qm"]) | (A["f"] <= 0); signerr = np.isfinite(A["Qm"]) & (np.sign(A["Qm"]) != np.sign(A["Qt"])) & ~fail
    ok = ~fail & ~signerr & (np.abs(A["Qt"]) > 0); e = np.log(np.abs(A["Qm"][ok]) / np.abs(A["Qt"][ok]))
    rsd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 3 else np.nan
    print(f"{len(files)} (junction, α, seed) runs, {len(rows)} edge measurements: profile-fit failures {fail.mean():.1%}, "
          f"sign errors {signerr.mean():.1%}")
    print(f"overall: median Q̂/Q_true ×{np.exp(np.median(e)):.3f}, robust SD of log ratio {rsd(e):.3f}")

    def table(key, edges):
        x = A[key][ok]; print(f"\nby {key}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x >= lo) & (x < hi); fa = (A[key] >= lo) & (A[key] < hi)
            if s.sum() > 5:
                print(f"  {lo:g}–{hi:g}: n {s.sum():4d}  bias ×{np.exp(np.median(e[s])):.3f}  scatter {rsd(e[s]):.3f}  "
                      f"fail {fail[fa].mean():.0%}  sign err {signerr[fa].mean():.0%}")
    table("v", [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 5]); table("Rseg", [0, 4, 6, 8, 11, 17])
    table("f", [0, 0.3, 0.5, 0.7, 0.9, 1.01]); table("align", [0, 0.6, 0.8, 0.9, 0.95, 1.01]); table("LR", [0, 2, 3, 4, 6, 50])
    c = A["c"][ok]
    print(f"\nreal reliability model on synthetic edges: Spearman(|e|, c) {spearmanr(np.abs(e), c)[0]:+.2f}")
    for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 9)):
        s = (c >= lo) & (c < hi)
        if s.sum() > 5:
            print(f"  c {lo}–{hi}: n {s.sum():4d}  robust SD of log ratio {rsd(e[s]):.3f}  (bias ×{np.exp(np.median(e[s])):.3f})")
    lv, lR = np.log(A["v"][ok]), np.log(A["Rseg"][ok]); X = np.column_stack([np.ones(ok.sum()), lv, lR])
    hub = lambda r: np.sum(np.where(np.abs(r) < 0.3, 0.5 * r ** 2, 0.3 * (np.abs(r) - 0.15)))
    b = minimize(lambda th: hub(e - X @ th), np.zeros(3), method="Nelder-Mead").x
    print(f"\nbias model: log(Q̂/Q) = {b[0]:+.3f} {b[1]:+.3f}·log v {b[2]:+.3f}·log R_seg  → at v 0.1/0.3/0.9 px/frame, R 8 px: "
          + " ".join(f"×{np.exp(b[0] + b[1] * np.log(v) + b[2] * np.log(8)):.2f}" for v in (0.1, 0.3, 0.9)))
    iv = np.array([x[0] for x in imb]); im = np.array([x[1] for x in imb])
    print("\njunction imbalance |ΣQ̂|/Σ|Q̂| (exact truth = 0):")
    for v in sorted(set(iv)):
        s = (iv == v) & np.isfinite(im)
        print(f"  parent v {v}: median {np.median(im[s]):.3f}  IQR {np.percentile(im[s], 25):.3f}–{np.percentile(im[s], 75):.3f}  (n {s.sum()})")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    axs[0].scatter(A["v"][ok], np.exp(e), s=4, alpha=0.3); axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].axhline(1, color="k")
    axs[0].set_xlabel("true branch speed (px/frame)"); axs[0].set_ylabel("Q̂ / Q_true")
    axs[1].scatter(c, np.abs(e), s=4, alpha=0.3); axs[1].set_xlabel("reliability scale c (real model)"); axs[1].set_ylabel("|log(Q̂/Q_true)|")
    axs[2].hist(im[np.isfinite(im)], bins=40, range=(0, 1)); axs[2].set_xlabel("junction imbalance (synthetic)")
    fig.tight_layout(); fig.savefig(fig_path, dpi=110); print("wrote", fig_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", default=str(paths.RESULTS / "benchmark"))
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--junctions", type=int, default=0, help="first N only (0 = all)")
    ap.add_argument("--reliability", default=str(paths.RESULTS / "reliability_model_v1.json"))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser() / "junctions"; out.mkdir(parents=True, exist_ok=True)
    nodes = Bundle(a.bundle).junctions(); nodes = nodes[:a.junctions] if a.junctions else nodes
    if not a.summary_only:
        jobs = [(n, al, s) for n in nodes for al in JUNCTION_ALPHAS for s in JUNCTION_SEEDS]; t0 = time.time()
        for i in range(0, len(jobs), 40):
            Parallel(n_jobs=a.jobs)(delayed(run_one)(a.bundle, out, *j) for j in jobs[i:i + 40])
            print(f"{min(i + 40, len(jobs))}/{len(jobs)} (junction × α × seed) done, {time.time() - t0:.0f} s", flush=True)
    files = sorted(f for f in out.glob("*.pkl") if int(f.name.split("_")[0]) in set(nodes))
    summarize(files, a.reliability, out.parent / "junctions_summary.png")


if __name__ == "__main__":
    main()
